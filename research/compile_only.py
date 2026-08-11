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

"""Compile a workload for Spyre with NO device, and dump what the compiler decided.

WHAT THIS IS FOR. The accelerator is dead, but almost none of the compiler needs it:
lowering, every pre-scheduling pass, LX scratchpad planning, the cost model, and SDSC
codegen are pure host-side graph transformation. The first thing that touches device memory
is `prepare_kernel` (`csrc/prepare_kernel.cpp:268`), which allocates program memory at
*compile* time -- long after the decisions this study is about. So we compile until that
throws, and keep everything produced before it.

Two things make it work, and both are load-bearing:

* `IS_INDUCTOR_SPAWNED_SUBPROCESS=1` (`torch_spyre/__init__.py:38-42`) marks the runtime
  pre-initialised, so `_lazy_init()` returns immediately. Without it, `torch.manual_seed`
  and `compile_fx` both call `start_runtime()` and die on a broken card.
* `FakeTensorMode` keeps inputs off the device entirely, replacing `.to("spyre")`, which
  would otherwise hit the real allocator.

`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` is not optional: `CustomPreSchedulingPasses.uuid()`
keys the Inductor cache, and a cache hit skips the passes AND therefore the dump.

    python3 research/compile_only.py --workload swiglu_mlp --tiles 8 --out feats.json

The JSON it writes is the same feature-vector shape the sweep harness records, so it feeds
straight into `research/lx_choice.py --records feats.json`.

UNVERIFIED. This has not been run -- the machine it was written on has no `torch_spyre._C`
and no SDK. Treat the first run as an experiment, and check the sanity line it prints.
"""

import argparse
import json
import os
import sys

# Must be set before torch_spyre is imported anywhere: __init__ reads it at module scope.
os.environ.setdefault("IS_INDUCTOR_SPAWNED_SUBPROCESS", "1")
os.environ.setdefault("SPYRE_DUMP_COST", "1")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
import torch._inductor.compile_fx as cfx  # noqa: E402
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402
from torch.fx.experimental.proxy_tensor import make_fx  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from workloads import WORKLOADS  # noqa: E402


def _sanity():
    """Report whether the no-device escape hatch actually holds on this machine."""
    import torch_spyre  # noqa: F401

    try:
        torch.manual_seed(1)
        seeded = True
    except Exception as exc:  # noqa: BLE001
        print(f"ESCAPE HATCH FAILED: manual_seed raised {type(exc).__name__}: {exc}")
        return False
    init = getattr(getattr(torch, "spyre", None), "is_initialized", lambda: "?")()
    print(f"escape hatch OK (manual_seed={seeded}, device initialized={init})")
    return True


def compile_workload(name, tiles, dtype=torch.float16):
    """Compile one workload on fake tensors; return the recorded features.

    The `prepare_kernel` failure at the end is EXPECTED and is not an error: the cost dump
    and the allocator's decisions are already recorded by then.
    """
    from torch_spyre._inductor import dump_cost_model as dcm

    fn, build = WORKLOADS[name](tiles=tiles, dtype=dtype)

    with FakeTensorMode(allow_non_fake_inputs=True):
        cpu_args = build()
        args = [torch.empty(a.shape, dtype=a.dtype, device="spyre") for a in cpu_args]
        gm = make_fx(fn)(*args)
        try:
            cfx.compile_fx(gm, list(args))
            print("compile completed (no device access was needed)")
        except Exception as exc:  # noqa: BLE001 -- expected at prepare_kernel
            print(f"stopped after codegen, as expected: {type(exc).__name__}: {exc}")

    feats = list(getattr(dcm, "LAST_FEATS", []) or [])
    io = dict(getattr(dcm, "LAST_IO", {}) or {})
    if not feats:
        print(
            "WARNING: no features recorded. Either the passes did not run (a cache hit -- "
            "check TORCHINDUCTOR_FORCE_DISABLE_CACHES) or compilation stopped earlier "
            "than expected."
        )
    return feats, io


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", default="swiglu_mlp", choices=sorted(WORKLOADS))
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--sanity", action="store_true", help="check the escape hatch and exit"
    )
    args = ap.parse_args()

    if args.sanity:
        return 0 if _sanity() else 1
    if not _sanity():
        return 1

    feats, io = compile_workload(args.workload, args.tiles)
    print(
        f"\n{len(feats)} ops recorded; HBM {io.get('hbm_bytes', 0) / 2**20:.1f} MB, "
        f"LX {io.get('lx_bytes', 0) / 2**20:.1f} MB"
    )
    for op in feats:
        d = op if isinstance(op, dict) else op.__dict__
        mems = [(a.get("name"), a.get("mem")) for a in (d.get("args") or [])]
        print(f"   {d.get('name'):<28} {mems}")

    if args.out:
        # Same envelope as the sweep database, so lx_choice.py can read it unchanged.
        rec = {
            "op": args.workload,
            "label": f"{args.workload} tiles={args.tiles}",
            "tiles": args.tiles,
            "kernel_us": 0.0,
            "failed": False,
            "feats": [op if isinstance(op, dict) else op.__dict__ for op in feats],
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"records": [rec]}, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
