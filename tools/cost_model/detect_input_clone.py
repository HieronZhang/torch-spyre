# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detect the graph-input LX clone in a recorded bundle, offline.

WHY. Upstream #3154 pins a shared graph input into LX by cloning it, which changes a
kernel's access pattern without changing its byte count. Measured on a 4096x2048 softmax:
261 us where the pre-clone database has 390 us, while the model predicts 320 us either way.
Decomposed, the kernel sits 38 us above the pure-bandwidth floor but ``alpha*min(R,W)``
charges 96 us -- 2.6x too much, because alpha prices read/write bus turnaround and the clone
removes most of the interleaving.

`alpha` was calibrated (report section 2) on plain pointwise ops, which have no shared input
and so never get the clone. Those points are still good. Re-fitting one alpha over both
regimes would fit neither, so scoring has to be SEGMENTED by whether a bundle was cloned --
which is what this detects.

No extractor change is needed: the clone is already an ordinary op in the recorded features.
It reads a graph input from HBM, writes to LX, and later ops read that LX buffer instead of
the input.

    python3 tools/cost_model/detect_input_clone.py            # scan the database
    python3 tools/cost_model/detect_input_clone.py FEATS.json # scan one feature dump
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import records_path  # noqa: E402


def _args(op):
    return op.get("args") or []


def clone_ops(feats):
    """Names of ops in ``feats`` that are graph-input LX clones.

    The shape to match, from a measured post-#3154 softmax bundle:

        Pointwise  in: arg0_1 (hbm)  out: op5 (lx)      <- the clone
        amax       in: buf5   (lx)   ...                <- reads the clone, not arg0_1

    Requiring a downstream LX consumer is what separates a clone from an ordinary op that
    happens to land its output in LX: a clone exists only to be read by something else.
    """
    out = []
    for i, op in enumerate(feats):
        ins = [a for a in _args(op) if a.get("role") == "input"]
        outs = [a for a in _args(op) if a.get("role") == "output"]
        if len(ins) != 1 or len(outs) != 1:
            continue  # a clone is strictly one-in one-out
        src, dst = ins[0], outs[0]
        if src.get("mem") != "hbm" or dst.get("mem") != "lx":
            continue
        if not str(src.get("name", "")).startswith("arg"):
            continue  # a graph input, not an intermediate
        if src.get("elems") != dst.get("elems"):
            continue  # a copy moves the whole tensor
        later = feats[i + 1 :]
        if any(
            a.get("mem") == "lx" and a.get("role") == "input"
            for o in later
            for a in _args(o)
        ):
            out.append(op.get("name", f"op{i}"))
    return out


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as fh:
            feats = json.load(fh)
        found = clone_ops(feats)
        print(f"clone ops: {found or 'none'}  ({len(feats)} ops in the bundle)")
        return 0

    with open(records_path(), encoding="utf-8") as fh:
        records = json.load(fh)["records"]
    cloned = plain = 0
    for r in records:
        feats = r.get("feats")
        if not feats:
            continue
        if clone_ops(feats):
            cloned += 1
        else:
            plain += 1
    total = cloned + plain
    print(f"records with features: {total}")
    print(f"  with a graph-input LX clone : {cloned}")
    print(f"  without                     : {plain}")
    if cloned == 0:
        print(
            "\nNone found, which is expected for a database collected before #3154.\n"
            "Re-sweep on a current build, then score the two groups separately."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
