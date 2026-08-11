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

"""Emit the exact LX allocations the hardware session should measure.

The session compares allocations, not solvers. Solvers are an indirect and unreliable way to
produce a chosen allocation -- two of them often agree, and then an equal runtime confirms
nothing. `LX_FORCE_ONLY` (see ``_lx_force_override`` in ``scratchpad/allocator.py``) pins the
set directly, so this writes the sets and their predicted times to JSON for the runner to
consume. Nothing is hand-typed: every set and every microsecond here comes from the same
enumeration that produced the claim.

For each flash configuration where LX capacity binds it emits four arms:

  optimum   the PROVEN time-optimal set over every feasible allocation
  cpsat     the exact argmax of CP-SAT's byte objective
  greedy    what the shipped default solver picks
  bestfit   what the (lifetime - discount)/uses ordering picks

    python3 research/emit_forced_allocations.py --records <db.json>
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "tools", "cost_model"))

import lx_choice as L  # noqa: E402
import lx_experiment as X  # noqa: E402

CAP = L.LX_CAPACITY_BYTES


def build_cases(records, op="flash_attn", min_regret=0.0):
    """Every contested configuration of ``op``, with each policy's allocation solved exactly.

    Importable so the session runner can regenerate the sets itself instead of depending on
    a file someone remembered to refresh -- the sets are only valid for the features they
    were derived from, and a re-sweep changes those.
    """
    cands = [
        r
        for r in records
        if r.get("op") == op and r.get("feats") and r.get("kernel_us")
    ]
    out = []
    for r in cands:
        feats = r["feats"]
        movable = L.bundle_intermediates(feats)
        if not movable:
            continue
        if L.peak_footprint(feats, set(movable)) <= CAP:
            continue  # capacity does not bind: every arm would be identical
        exact, evaluate, n_eval = L.exhaustive_policies(feats, CAP)
        if n_eval < 2:
            continue  # only the empty allocation fits: nothing to choose between
        seq = X.solver_allocations(feats)
        arms = {
            "optimum": exact["time"],
            "cpsat": exact["cpsat"],
            "greedy": seq["greedy"],
            "bestfit": seq["bestfit"],
        }
        ref = evaluate(exact["time"])
        label = (r.get("label") or "").split(" (IR")[0].split(" [was")[0]
        entry = {
            "label": label,
            "measured_us": r.get("kernel_us"),
            "n_movable": len(movable),
            "n_feasible": n_eval,
            "env": _env_from_record(r),
            "bench_op": r.get("op") or op,
            "arms": {},
        }
        for name, s in arms.items():
            entry["arms"][name] = {
                "lx": sorted(s),
                "n_lx": len(s),
                "pred_us": evaluate(s),
                "predicted_regret_pct": (evaluate(s) - ref) / ref * 100.0,
            }
        # Arms that name the SAME set cannot be told apart by a measurement; say so here
        # rather than discovering it after the hardware time is spent.
        seen: dict = {}
        for name, a in entry["arms"].items():
            seen.setdefault(tuple(a["lx"]), []).append(name)
        entry["distinct_allocations"] = len(seen)
        entry["duplicate_arms"] = [v for v in seen.values() if len(v) > 1]
        out.append(entry)
    return out


def emit(records_path, out_path, op="flash_attn", quiet=False, top=0):
    """Solve, write the JSON, and describe what was found. Returns the case list."""
    with open(records_path, encoding="utf-8") as fh:
        records = json.load(fh)["records"]
    cases = build_cases(records, op)
    # Most contested first, so a capped session spends its runs where the spread is.
    cases.sort(
        key=lambda c: -max(a["predicted_regret_pct"] for a in c["arms"].values())
    )
    if top:
        cases = cases[:top]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"cap_bytes": CAP, "cases": cases}, fh, indent=2)
    if not quiet:
        describe(cases, op, out_path)
    return cases


def describe(out, op, out_path):
    """Print the arms, and say plainly where a measurement cannot separate them."""
    print(f"{len(out)} contested {op} configuration(s) -> {out_path}\n")
    for e in out:
        print(f"  {e['label']}")
        print(
            f"    measured {e['measured_us']:.0f} us | {e['n_movable']} movable | "
            f"{e['n_feasible']:,} feasible | "
            f"{e['distinct_allocations']} distinct arm(s)"
        )
        for name, a in e["arms"].items():
            print(
                f"      {name:<9} keeps {a['n_lx']:>2}  "
                f"pred {a['pred_us']:>10.1f} us  "
                f"regret {a['predicted_regret_pct']:>+6.1f}%"
            )
        if e["duplicate_arms"]:
            for grp in e["duplicate_arms"]:
                print(
                    f"      NOTE: {' and '.join(grp)} name the SAME set -- "
                    f"they must measure alike"
                )
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="")
    ap.add_argument("--op", default="flash_attn")
    ap.add_argument(
        "--top",
        type=int,
        default=0,
        help="keep only the N most contested cases (0 = all)",
    )
    ap.add_argument("--out", default=os.path.join(_HERE, "forced_allocations.json"))
    args = ap.parse_args()

    import eval_model as E

    emit(args.records or E.records_path(), args.out, args.op, top=args.top)
    return 0


def _env_from_record(r):
    """The environment that reproduces a recorded configuration.

    Built from the record's STRUCTURED fields (``rows``/``cols``/``tiles``/``cores``) where
    they exist, and only falling back to parsing the label for flash, whose FA_* knobs are
    not stored as columns. Parsing prose is the last resort, not the first.
    """
    env = {}
    if r.get("rows"):
        env["BENCH_ROWS"] = str(int(r["rows"]))
    if r.get("cols"):
        env["BENCH_COLS"] = str(int(r["cols"]))
    if r.get("tiles") is not None:
        env["BENCH_TILES"] = str(int(r["tiles"]))
    if r.get("cores"):
        env["SENCORES"] = str(int(r["cores"]))
    if r.get("lx") is not None:
        env["LX_PLANNING"] = str(int(r["lx"]))
    if (r.get("op") or "") == "flash_attn":
        env.update(_env_from_label(r.get("label") or ""))
        env.pop("BENCH_ROWS", None)
        env.pop("BENCH_COLS", None)
        env.pop("BENCH_TILES", None)
    return env


def _env_from_label(label):
    """The FA_* environment that reproduces a recorded flash configuration."""
    env = {}
    for key, pat in (
        ("FA_H", "H="),
        ("FA_LQ", "Lq="),
        ("FA_LK", "Lk="),
        ("FA_D", "D="),
    ):
        for tok in label.split():
            if tok.startswith(pat):
                env[key] = tok[len(pat) :]
    for key, pat in (
        ("FA_H_TILES", "htiles="),
        ("FA_LQ_TILES", "qtiles="),
        ("FA_LK_TILES", "ktiles="),
        ("FA_B_TILES", "btiles="),
    ):
        for tok in label.split():
            if tok.startswith(pat):
                env[key] = tok[len(pat) :]
    return env


if __name__ == "__main__":
    raise SystemExit(main())
