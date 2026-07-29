#!/usr/bin/env python3
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
"""Probe ONE flash-attention hint configuration: is it even runnable, and if so how fast?

Why this exists: most of the cat-7 flash sweep "did not finish". There are two DISTINCT
causes, and only one of them is a compiler error:

  (a) LX PRESSURE -> the region SPILLS to HBM and runs SLOWER (it does NOT hard-fail).
      The fused flash region holds per-tile intermediates (`scores`, `exp_scores`, ... each
      [B_t,H_t,Lq_t,Lk_t] fp16); the LX layout solver is sized 2 MB * (1 - DXP_LX_FRAC_AVAIL)
      = ~1638 KB per core by default (scratchpad/allocator.py:449). What does not fit is placed
      in HBM -- exactly the `lx_spill` term the cost model carries. So this estimate predicts
      SLOWDOWN, not failure.
      CALIBRATION (important): the shipped flash_attn_example.py -- which RUNS -- uses
      h_block=4, q_block=Lq//4, kv_block=Lk (i.e. H_TILES=8, LQ_TILES=4, LK_TILES=1 at
      Lq=Lk=4096) and scores that estimate to ~2048 KB/core. So ~2 MB/core is EMPIRICALLY
      RUNNABLE, and the estimate below should be read as relative pressure, not a hard limit.
      Shrink the per-core set with smaller coarse tiles and/or a work_div that gets PLACED --
      but note MORE TILES ALSO MEANS A BIGGER UNROLLED PROGRAM (config.unroll_loops=1), and the
      example's own comments call out block size as a COMPILE-TIME driver. That trade-off
      (LX pressure vs compile time) is the thing worth measuring.
  (b) INVALID work_div split -> a fast, hard InductorError, e.g.
        "work_division_hint: buf5 dim d0 size=2 is not evenly divisible by split=4"
      raised in work_division.py::_apply_user_hint. The rule (confirmed from the failures):
      a work_div split on a named dim must divide that dim's PER-TILE size, i.e.
      (dim / tiles) % split == 0.  The two observed failures fit exactly:
        H=32, H_TILES=16 -> per-tile H = 2, split H:4  -> FAIL
        H=32, H_TILES=8  -> per-tile H = 4, split H:8  -> FAIL

  Note on work_div PRODUCT > 32: by itself it is NOT an error -- the pass silently SKIPS a
  split it cannot place (verified: 10 product-256 configs compiled fine). But a skipped split
  means the tile is NOT divided across cores, so the per-core LX working set stays huge -> it
  feeds straight back into cause (a). That is the link between the two symptoms.

Additionally, a run that hits (a) can HANG in Spyre-runtime teardown after printing its error,
so each config must be probed in its OWN process under `timeout` -- see run_flash_probe.sh.

Usage:
    # pure arithmetic, no device needed -- check a config before spending a compile on it:
    python3 flash_probe.py --validate-only --h-tiles 16 --wd H:4,Lq:8

    # full probe on the Spyre machine (validate, then compile, then time it):
    python3 flash_probe.py --h-tiles 8 --lq-tiles 4 --wd H:4,Lq:4

Knobs (flags or FA_* env): B/H/LQ/LK/D shape, B_TILES/H_TILES/LQ_TILES/LK_TILES coarse tile
COUNTS, WD the work_div hint ("H:4,Lq:8,Lk:8"). Prints exactly one `RESULT: ...` line so a
driver script can classify the outcome.
"""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--b", type=int, default=int(os.environ.get("FA_B", "1")))
    p.add_argument("--h", type=int, default=int(os.environ.get("FA_H", "32")))
    p.add_argument("--lq", type=int, default=int(os.environ.get("FA_LQ", "2048")))
    p.add_argument("--lk", type=int, default=int(os.environ.get("FA_LK", "2048")))
    p.add_argument("--d", type=int, default=int(os.environ.get("FA_D", "128")))
    p.add_argument("--b-tiles", type=int, default=int(os.environ.get("FA_B_TILES", "1")))
    p.add_argument("--h-tiles", type=int, default=int(os.environ.get("FA_H_TILES", "8")))
    p.add_argument("--lq-tiles", type=int, default=int(os.environ.get("FA_LQ_TILES", "4")))
    p.add_argument("--lk-tiles", type=int, default=int(os.environ.get("FA_LK_TILES", "1")))
    p.add_argument("--wd", default=os.environ.get("FA_WD", "H:4,Lq:8,Lk:8"))
    p.add_argument("--lx-per-core", type=float, default=float(os.environ.get("LX_PER_CORE", str(int((2 << 20) * 0.8)))),
                   help="LX budget/core = 2MB*(1-DXP_LX_FRAC_AVAIL); default 1638 KB")
    p.add_argument("--variant", choices=["online", "functional"], default="online",
                   help="online = the example's form (pre-allocated output/real_max/denominator "
                        "mutated with .copy_() -> 7 non-intermediate tensors, EXCEEDS the "
                        "5-tensor SDSC bundle limit for an atomic coarse-tiled loop). "
                        "functional = the same math with no pre-allocated accumulators "
                        "(q,k,v,mask,out = 5) -- valid because Lk is not coarse-tiled, so the "
                        "online-softmax carry is vestigial.")
    p.add_argument("--validate-only", action="store_true", help="no device needed")
    p.add_argument("--reps", type=int, default=int(os.environ.get("REPS", "3")))
    p.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "2")))
    return p.parse_args()


def validate(a):
    """Pure arithmetic. Returns (ok, [problems], [warnings], info) -- no torch, no device."""
    wd = {}
    for part in a.wd.split(","):
        part = part.strip()
        if not part:
            continue
        nm, val = part.split(":")
        wd[nm.strip()] = int(val)
    dims = {"B": (a.b, a.b_tiles), "H": (a.h, a.h_tiles), "Lq": (a.lq, a.lq_tiles), "Lk": (a.lk, a.lk_tiles)}

    problems, warns = [], []
    # 1. coarse tile count must divide its dim
    for nm, (size, tiles) in dims.items():
        if tiles < 1:
            problems.append(f"{nm}_TILES={tiles} < 1")
        elif size % tiles:
            problems.append(f"{nm}={size} not divisible by {nm}_TILES={tiles}")
    # 2. THE killer: a work_div split must divide the PER-TILE size of that dim
    for nm, split in wd.items():
        if nm not in dims:
            warns.append(f"work_div dim {nm!r} is not one of B/H/Lq/Lk -- ignored by the hint")
            continue
        size, tiles = dims[nm]
        per_tile = size // tiles if tiles else 0
        if per_tile == 0 or split > per_tile or per_tile % split:
            problems.append(
                f"work_div {nm}:{split} does not divide per-tile {nm} = {size}/{tiles} = {per_tile}"
                "  -> InductorError 'not evenly divisible'"
            )
    # 3. over-subscription: not an error by itself, but the split is silently SKIPPED, which
    #    leaves the per-core LX working set undivided -> feeds cause (a).
    prod = 1
    for v in wd.values():
        prod *= v
    placed = prod  # cores the split can actually be placed on
    if prod > 32:
        placed = 1  # conservative: assume the over-subscribed splits are skipped entirely
        warns.append(f"work_div product = {prod} > 32 cores -- the pass SILENTLY SKIPS splits it "
                     "cannot place, so the tile may NOT be divided across cores (this is what "
                     "blows up the per-core LX working set below)")

    # 4. THE dominant failure mode: per-core LX scratchpad working set.
    #    The fused region holds per-tile intermediates; scores/exp_scores dominate, each
    #    [B_t,H_t,Lq_t,Lk_t] fp16. Only ~512 KB/core is practically available.
    per_tile = {nm: (size // tiles if tiles else 0) for nm, (size, tiles) in dims.items()}
    tile_elems = per_tile["B"] * per_tile["H"] * per_tile["Lq"] * per_tile["Lk"]
    live = 2 * tile_elems * 2  # >= scores + exp_scores, fp16 (a LOWER bound on the live set)
    # We do NOT know exactly how many cores the tile lands on: an explicit, placeable work_div
    # divides it, and absent that the automatic work-division pass still spreads it. So bound it:
    #   BEST  = all 32 cores share the tile
    #   WORST = only the (placeable) hinted split divides it
    lx_best = live / 32.0
    lx_worst = live / max(1, min(32, placed))
    if lx_best > a.lx_per_core:
        warns.append(
            f"HIGH LX PRESSURE: even spread over all 32 cores the tile needs "
            f"~{lx_best / 1024:.0f} KB/core > {a.lx_per_core / 1024:.0f} KB budget -> expect SPILL "
            "to HBM (slower), not a failure. For scale, the shipped flash_attn_example.py runs at "
            "~2048 KB/core by this same estimate.")
    elif lx_worst > a.lx_per_core:
        warns.append(
            f"LX pressure depends on placement: {lx_worst / 1024:.0f} KB/core if only the hint "
            f"divides the tile, {lx_best / 1024:.0f} KB/core if it spreads over 32. Prefer a "
            "work_div with product <= 32 that divides the per-tile dims, so the division is "
            "guaranteed.")
    scores_gb = a.b * a.h * a.lq * a.lk * 2 / 1e9
    info = {
        "per_tile": per_tile, "wd": wd, "wd_product": prod, "scores_gb": scores_gb,
        "lx_best": lx_best, "lx_worst": lx_worst, "live_tile_bytes": live, "placed_cores": placed,
    }
    return (not problems), problems, warns, info


def build_and_run(a):
    """Compile + time the flash program. Self-contained (does NOT import profile_ops.py)."""
    import math
    import statistics

    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
    import torch
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch.profiler import ProfilerActivity, profile
    from torch_spyre._inductor import spyre_hint

    dev = torch.device("spyre")
    B, H, Lq, Lk, D = a.b, a.h, a.lq, a.lk, a.d
    wd = {p.split(":")[0].strip(): int(p.split(":")[1]) for p in a.wd.split(",") if p.strip()}
    scale = 1.0 / math.sqrt(math.sqrt(D))

    def declare():
        pnd.declare_tensor_dim("B", B)
        pnd.declare_tensor_dim("H", H)
        pnd.declare_tensor_dim("Lq", Lq)
        pnd.declare_tensor_dim("Lk", Lk)
        pnd.declare_tensor_dim("D", D)

    declare()

    def flash_functional(queries, keys, values, mask):
        """Same result, but NO pre-allocated output/real_max/denominator and no .copy_():
        only q,k,v,mask + the returned output are non-intermediate -> 5 tensors, which fits
        the atomic coarse-tiled bundle budget that the 'online' form blows."""
        pnd.name_tensor_dims(queries, ["B", "H", "Lq", "D"])
        pnd.name_tensor_dims(keys, ["B", "H", "Lk", "D"])
        pnd.name_tensor_dims(values, ["B", "H", "Lk", "D"])
        pnd.name_tensor_dims(mask, ["B", "H", "Lq", "Lk"])
        with spyre_hint(tiles={"B": a.b_tiles}), spyre_hint(tiles={"H": a.h_tiles}), \
             spyre_hint(tiles={"Lq": a.lq_tiles}), spyre_hint(tiles={"Lk": a.lk_tiles}), \
             spyre_hint(work_div=wd):
            keys_t = (keys * scale).transpose(-1, -2)
            scores = torch.matmul(queries * scale, keys_t) + mask
            m = torch.amax(scores, dim=-1, keepdim=True)
            e = torch.exp(scores - m)
            return torch.matmul(e / e.sum(dim=-1, keepdim=True), values)

    def flash(queries, keys, values, mask):
        pnd.name_tensor_dims(queries, ["B", "H", "Lq", "D"])
        pnd.name_tensor_dims(keys, ["B", "H", "Lk", "D"])
        pnd.name_tensor_dims(values, ["B", "H", "Lk", "D"])
        pnd.name_tensor_dims(mask, ["B", "H", "Lq", "Lk"])
        output = torch.zeros_like(queries)
        real_max = torch.full((B, H, Lq, 64), float("-inf"), device=queries.device,
                              dtype=torch.float16).amax(dim=-1)
        denominator = torch.zeros((B, H, Lq, 64), device=queries.device,
                                  dtype=torch.float16).amax(dim=-1)
        with spyre_hint(tiles={"B": a.b_tiles}), spyre_hint(tiles={"H": a.h_tiles}), \
             spyre_hint(tiles={"Lq": a.lq_tiles}), spyre_hint(tiles={"Lk": a.lk_tiles}), \
             spyre_hint(work_div=wd):
            keys_t = (keys * scale).transpose(-1, -2)
            scores = torch.matmul(queries * scale, keys_t) + mask
            block_max = torch.amax(scores, dim=-1)
            running_max = torch.maximum(real_max, block_max)
            exp_scores = torch.exp(scores - running_max.unsqueeze(-1))
            correction = torch.exp(real_max - running_max)
            denominator.copy_(denominator * correction + exp_scores.sum(dim=-1))
            output.copy_(output * correction.unsqueeze(-1) + torch.matmul(exp_scores, values))
            real_max.copy_(running_max)
        return output

    q = torch.rand(B, H, Lq, D, dtype=torch.float16).to(dev)
    k = torch.rand(B, H, Lk, D, dtype=torch.float16).to(dev)
    v = torch.rand(B, H, Lk, D, dtype=torch.float16).to(dev)
    m = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16).to(dev)  # broadcast mask, as in the example
    compiled = torch.compile(flash_functional if a.variant == 'functional' else flash)

    for _ in range(a.warmup):  # first call compiles -> this is where an InductorError fires
        declare()
        compiled(q, k, v, m).cpu()
    times = []
    for _ in range(a.reps):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            declare()
            compiled(q, k, v, m).cpu()
        us = 0.0
        for ev in prof.key_averages():
            t = getattr(ev, "self_device_time_total", 0) or 0
            name = ev.key or ""
            if t > 0 and "Memset" not in name and "Memcpy" not in name:
                us += t
        if us > 0:
            times.append(us)
    return statistics.median(times) if times else float("nan")


def main():
    a = parse_args()
    tag = (f"H{a.h}_Lq{a.lq}_Lk{a.lk}_D{a.d}_h{a.h_tiles}q{a.lq_tiles}k{a.lk_tiles}"
           f"_wd[{a.wd}]_{a.variant}")
    ok, problems, warns, info = validate(a)

    print(f"CONFIG  {tag}")
    print(f"  per-tile sizes: {info['per_tile']}   work_div={info['wd']} (product {info['wd_product']})")
    print(f"  scores/mask    : {info['scores_gb']:.3f} GB each  ([B,H,Lq,Lk] fp16)")
    print(f"  LX working set : ~{info['lx_best'] / 1024:.0f} KB/core at 32 cores, "
          f"~{info['lx_worst'] / 1024:.0f} KB/core if only the hint divides "
          f"(live tile {info['live_tile_bytes'] / 1024:.0f} KB)"
          f"   [available ~{a.lx_per_core / 1024:.0f} KB]")
    for w in warns:
        print(f"  WARN  {w}")
    for pr in problems:
        print(f"  ERROR {pr}")

    if not ok:
        print(f"RESULT: INVALID  {tag}  ({problems[0]})")
        return 2
    if a.validate_only:
        print(f"RESULT: VALID  {tag}  (not compiled; --validate-only)")
        return 0
    try:
        us = build_and_run(a)
    except Exception as e:  # noqa: BLE001 -- we want the class + message for classification
        msg = str(e).replace("\n", " ")[:200]
        kind = "SPLIT-ERROR" if "evenly divisible" in msg else "ERROR"
        print(f"RESULT: {kind}  {tag}  ({type(e).__name__}: {msg})")
        return 3
    print(f"RESULT: OK  {tag}  kernel_us={us:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
