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

"""Am I on the same broken card, or a different one?

WHY. A Spyre card failed DDR initialisation on 2026-08-07 and every run since has died in
``start_runtime()``. Deleting the pod and creating a new one did NOT help, and the reason was
visible in the error payload: the RAS record carries the DDR calibration result, and its read
and write **eye diagrams are physical characteristics of that specific card**. All five
calibration rows from the replacement pod appeared verbatim in the original log, so the
scheduler had handed back the same card -- deleting the pod merely released it to the pool.

The workaround is to create a second pod WITHOUT deleting the first, so the bad card stays
allocated and the new pod is forced onto a different one. This script checks whether that
worked, instead of assuming it did.

    python3 research/card_fingerprint.py

Exit status: 0 the device initialises, 2 a DIFFERENT card is also failing, 3 the SAME card.
"""

import regex as re

#: Distinct values seen per column across 13,890 calibration rows from the failing card
#: (docs/source/user_guide/examples/cost_model_sweep_20260807_020123.log). Eye margins are
#: per-card; LPDDR_Fail is included but carries less weight, since a shared failure mode
#: could plausibly produce similar retry counts on any card.
KNOWN_BAD = {
    "RD_eye_min": {26, 27, 28},
    "RD_eye_avg": {37},
    "RD_eye_max": {42, 43, 44},
    "WR_eye_min": {31, 32, 33, 34},
    "WR_eye_avg": {39},
    "WR_eye_max": {45, 46, 47},
    "LPDDR_Fail": {1326, 1343, 1356, 1360},
}
_COLS = [
    "Cal_Err",
    "MRR_Pass",
    "RD_eye_min",
    "RD_eye_avg",
    "RD_eye_max",
    "WR_eye_min",
    "WR_eye_avg",
    "WR_eye_max",
    "LPDDR_Fail",
]
_ROW = re.compile(r"(?:\\n|\n)\s*" + r"\s+".join([r"(\d+)"] * 9))


def parse_rows(text):
    """Calibration rows out of a RAS payload, as dicts. Empty if the text has none."""
    return [dict(zip(_COLS, (int(v) for v in m))) for m in _ROW.findall(text)]


def verdict(rows):
    """Does this look like the card that failed on 2026-08-07?"""
    if not rows:
        return None, "no calibration rows in the error payload"
    checked = [c for c in KNOWN_BAD if any(c in r for r in rows)]
    inside = {c: all(r[c] in KNOWN_BAD[c] for r in rows if c in r) for c in checked}
    eyes = [c for c in checked if "eye" in c]
    same_eyes = eyes and all(inside[c] for c in eyes)
    return same_eyes, inside


def main():
    try:
        import torch

        import torch_spyre  # noqa: F401

        torch.manual_seed(1)
    except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement
        text = str(exc)
        rows = parse_rows(text)
        same, detail = verdict(rows)
        print(f"device init FAILED: {type(exc).__name__}")
        if not rows:
            print(
                "  no DDR calibration data in the payload -- a different failure mode."
            )
            print(f"  {text[:400]}")
            return 2
        print(f"\n{len(rows)} calibration rows reported:")
        for c in _COLS[2:]:
            vals = sorted({r[c] for r in rows if c in r})
            mark = "matches" if detail.get(c) else "DIFFERS from"
            print(f"   {c:<11} {str(vals):<24} {mark} the known-bad card")
        if same:
            print(
                "\nSAME CARD as the one that failed on 2026-08-07. The eye margins are"
                "\nper-card physical characteristics and they match. A second pod did not"
                "\nget you different silicon -- ask for a different NODE, or have the admin"
                "\nmark this card unschedulable."
            )
            return 3
        print(
            "\nDIFFERENT CARD, and it is also failing DDR init. Two cards failing the same"
            "\nway points at the node -- driver state, firmware, or the host device manager"
            "\n-- rather than the silicon. Worth telling the admin explicitly."
        )
        return 2

    dev = getattr(torch, "spyre", None)
    print("device init OK.")
    for name in ("device_count", "is_initialized"):
        fn = getattr(dev, name, None)
        if callable(fn):
            try:
                print(f"   {name}: {fn()}")
            except Exception as exc:  # noqa: BLE001
                print(f"   {name}: unavailable ({type(exc).__name__})")
    print("\nYou have a working card. Re-run the sweep with:")
    print(
        "   python3 docs/source/user_guide/examples/run_cost_model_sweep.py "
        "--skip-measured"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
