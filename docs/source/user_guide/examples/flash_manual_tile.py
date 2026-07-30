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
"""Flash-attention design space, direction 2: EXPLICIT Python-level tiling + LX allocation.

Direction 1 (flash_probe.py / run_flash_probe.sh) asks the compiler to tile, via
`spyre_hint(tiles=..., work_div=...)`, and we discovered its main failure mode is LX
scratchpad exhaustion: the monolithic program materialises `scores`/`exp_scores` of
[B,H,Lq,Lk] and the per-tile slice still has to fit in LX.

Direction 2 (this file) writes the blocking OURSELVES, in Python. The classic flash loop
keeps only a [Bq, Bk] score block live at a time, with an online-softmax running (max,
denominator, accumulator) carried across key blocks. The working set is then bounded BY
CONSTRUCTION -- we choose Bq/Bk -- instead of hoping the planner finds a fit.

MODES (--mode):
  hint         monolithic program + spyre_hint(tiles/work_div)      [direction 1 baseline]
  manual-fused explicit Python block loops, the WHOLE nest inside one torch.compile
               (Inductor sees one big unrolled graph -- cf. `softmax_unrolled`)
  manual-sep   the same block loops, but the inner block step is its OWN compiled kernel,
               called once per (i, j) block (cf. `add3_sep`: dependency kept, fusion removed)

LX ALLOCATION KNOBS (direction 2's second half -- "how to allocate LX to which tensors").
These are real compiler options, applied before/at compile time:
  --lx-frac F    DXP_LX_FRAC_AVAIL: the layout solver is sized `2 MB * (1 - F)` per core
                 (scratchpad/allocator.py:449). Default F=0.2 -> a 1.6 MB budget. LOWER F
                 = MORE LX for our tensors.
  --lx-all       allow_all_ops_in_lx_planning: by default only a whitelist of op outputs
                 (max/amax/sum/exp/sub/mul/add/mm/bmm/div/... -- scratchpad/utils.py) may
                 live in LX; this widens it to every intermediate.
  --lx-boundary  LX_BOUNDARY_CLONES: insert clones at graph in/out boundaries so INPUT and
                 OUTPUT tensors can be LX-pinned too (off by default; known not correct for
                 every op type -- that is exactly what we are testing).
  --lx-solver S  layout_solver: greedy | bestfit | firstfit -- the packing algorithm, i.e.
                 whether a given set of buffers fits at all.
  --no-lx        LX_PLANNING=0: scratchpad off entirely (everything through HBM) -- the
                 control that tells you how much LX is buying you.

Correctness is checked against a reference attention on every run, so a fast-but-wrong
variant cannot be mistaken for a win.

    # verify the blocking math locally, no accelerator needed:
    python3 flash_manual_tile.py --device cpu --mode manual-fused --h 4 --lq 128 --lk 128

    # on the Spyre machine, compare the three programs at a fixed shape:
    for m in hint manual-fused manual-sep; do python3 flash_manual_tile.py --mode $m; done

    # ... and sweep the LX allocation for the manual program:
    python3 flash_manual_tile.py --mode manual-fused --lx-frac 0.05 --lx-all
"""
import argparse
import os
import statistics
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["hint", "manual-fused", "manual-sep"], default="manual-fused")
    p.add_argument("--device", choices=["spyre", "cpu"], default="spyre")
    p.add_argument("--b", type=int, default=1)
    p.add_argument("--h", type=int, default=32)
    p.add_argument("--lq", type=int, default=2048)
    p.add_argument("--lk", type=int, default=2048)
    p.add_argument("--d", type=int, default=128)
    # manual blocking
    p.add_argument("--bq", type=int, default=256, help="query block (rows of scores kept live)")
    p.add_argument("--bk", type=int, default=512, help="key block (cols of scores kept live)")
    p.add_argument("--bh", type=int, default=4, help="head block")
    # hint mode
    p.add_argument("--h-tiles", type=int, default=8)
    p.add_argument("--lq-tiles", type=int, default=8)
    p.add_argument("--lk-tiles", type=int, default=2)
    p.add_argument("--wd", default="H:2,Lq:4")
    # LX allocation
    p.add_argument("--lx-frac", type=float, default=None, help="DXP_LX_FRAC_AVAIL (default 0.2)")
    p.add_argument("--lx-all", action="store_true", help="allow_all_ops_in_lx_planning")
    p.add_argument("--lx-boundary", action="store_true", help="LX_BOUNDARY_CLONES (pin graph in/out)")
    p.add_argument("--lx-solver", choices=["greedy", "bestfit", "firstfit"], default=None)
    p.add_argument("--no-lx", action="store_true", help="LX_PLANNING=0")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--no-mask", action="store_true",
                   help="drop the additive mask entirely (it is all-zeros in this benchmark). "
                        "Also removes the slice of the mask's LAST dim, which is the stick dim "
                        "-- a prime suspect for 'no mechanism to resolve stick incompatibility'.")
    p.add_argument("--no-check", action="store_true", help="skip the correctness check")
    p.add_argument("--plan-only", action="store_true",
                   help="print the blocking plan + LX working set and exit (allocates nothing)")
    return p.parse_args()


A = parse_args()

# --- LX knobs that must be set BEFORE torch_spyre is imported (read at import/compile) ---
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
if A.lx_frac is not None:
    os.environ["DXP_LX_FRAC_AVAIL"] = str(A.lx_frac)
if A.lx_boundary:
    os.environ["LX_BOUNDARY_CLONES"] = "1"
if A.no_lx:
    os.environ["LX_PLANNING"] = "0"

import torch  # noqa: E402

SPYRE = A.device == "spyre"
if SPYRE:
    import torch_spyre  # noqa: F401,E402  -- registers the "spyre" device
    import torch_spyre._inductor.config as spyre_config  # noqa: E402
    import torch_spyre._inductor.propagate_named_dims as pnd  # noqa: E402
    from torch_spyre._inductor import spyre_hint  # noqa: E402

    if A.lx_all:
        spyre_config.allow_all_ops_in_lx_planning = True
    if A.lx_solver:
        spyre_config.layout_solver = A.lx_solver
    if A.no_lx:
        spyre_config.lx_planning = False

DEV = torch.device("spyre") if SPYRE else torch.device("cpu")
DT = torch.float16 if SPYRE else torch.float32  # fp32 on CPU so the check is meaningful
SCALE = A.d ** -0.25  # applied to BOTH q and k, as in flash_attn_example -> net 1/sqrt(D)


# ===========================================================================
# The programs
# ===========================================================================
def reference(q, k, v, mask):
    """Plain attention, used only to validate the blocked variants."""
    s = torch.matmul(q * SCALE, (k * SCALE).transpose(-1, -2))
    if mask is not None:
        s = s + mask
    return torch.matmul(torch.softmax(s.float(), dim=-1).to(v.dtype), v)


def block_step(q_i, k_j, v_j, mask_ij, m, ell, acc):
    """ONE (query-block, key-block) step of online-softmax flash attention.

    Live tensors here are only [.., Bq, Bk] (`s`, `p`) and [.., Bq, D] (`acc`) -- this is
    the whole point: the working set is set by Bq/Bk, not by Lq/Lk."""
    s = torch.matmul(q_i * SCALE, (k_j * SCALE).transpose(-1, -2))            # [B,H,Bq,Bk]
    if mask_ij is not None:
        s = s + mask_ij
    m_new = torch.maximum(m, s.amax(dim=-1))                                   # [B,H,Bq]
    p = torch.exp(s - m_new.unsqueeze(-1))
    alpha = torch.exp(m - m_new)
    ell = ell * alpha + p.sum(dim=-1)
    acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, v_j)                     # [B,H,Bq,D]
    return m_new, ell, acc


def manual_flash(q, k, v, mask, step):
    """Explicit Python blocking over (H, Lq, Lk). `step` is the per-block function --
    either the plain one (fused into the enclosing graph) or a separately-compiled one."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    outs = []
    for h0 in range(0, H, A.bh):
        h1 = min(h0 + A.bh, H)
        row = []
        for i0 in range(0, Lq, A.bq):
            i1 = min(i0 + A.bq, Lq)
            q_i = q[:, h0:h1, i0:i1, :]
            shp = q_i.shape[:-1]
            m = torch.full(shp, float("-inf"), device=q.device, dtype=q.dtype)
            ell = torch.zeros(shp, device=q.device, dtype=q.dtype)
            acc = torch.zeros_like(q_i)
            for j0 in range(0, Lk, A.bk):
                j1 = min(j0 + A.bk, Lk)
                m, ell, acc = step(
                    q_i, k[:, h0:h1, j0:j1, :], v[:, h0:h1, j0:j1, :],
                    None if mask is None else mask[:, :, i0:i1, j0:j1], m, ell, acc,
                )
            row.append(acc / ell.unsqueeze(-1))
        outs.append(torch.cat(row, dim=2))
    return torch.cat(outs, dim=1)


def build(q, k, v, mask):
    """Return the callable to time, for the selected --mode."""
    if A.mode == "hint":
        if not SPYRE:
            raise SystemExit("--mode hint needs --device spyre (it uses spyre_hint)")
        B, H, Lq, D = q.shape
        Lk = k.shape[2]
        wd = {p.split(":")[0].strip(): int(p.split(":")[1]) for p in A.wd.split(",") if p.strip()}

        def prep():
            pnd.declare_tensor_dim("B", B)
            pnd.declare_tensor_dim("H", H)
            pnd.declare_tensor_dim("Lq", Lq)
            pnd.declare_tensor_dim("Lk", Lk)
            pnd.declare_tensor_dim("D", D)

        prep()

        def flash(qq, kk, vv, mm):
            pnd.name_tensor_dims(qq, ["B", "H", "Lq", "D"])
            pnd.name_tensor_dims(kk, ["B", "H", "Lk", "D"])
            pnd.name_tensor_dims(vv, ["B", "H", "Lk", "D"])
            pnd.name_tensor_dims(mm, ["B", "H", "Lq", "Lk"])
            out = torch.zeros_like(qq)
            rmax = torch.full((B, H, Lq, 64), float("-inf"), device=qq.device, dtype=qq.dtype).amax(-1)
            den = torch.zeros((B, H, Lq, 64), device=qq.device, dtype=qq.dtype).amax(-1)
            with spyre_hint(tiles={"H": A.h_tiles}), spyre_hint(tiles={"Lq": A.lq_tiles}), \
                 spyre_hint(tiles={"Lk": A.lk_tiles}), spyre_hint(work_div=wd):
                s = torch.matmul(qq * SCALE, (kk * SCALE).transpose(-1, -2)) + mm
                bmax = torch.amax(s, dim=-1)
                rnew = torch.maximum(rmax, bmax)
                es = torch.exp(s - rnew.unsqueeze(-1))
                corr = torch.exp(rmax - rnew)
                den.copy_(den * corr + es.sum(dim=-1))
                out.copy_(out * corr.unsqueeze(-1) + torch.matmul(es, vv))
                rmax.copy_(rnew)
            return out

        return torch.compile(flash), prep
    if A.mode == "manual-fused":
        # one graph containing the whole (unrolled) block nest
        return torch.compile(lambda *a: manual_flash(*a, step=block_step)), None
    # manual-sep: each block step is its own compiled kernel, driven by a Python loop
    step = torch.compile(block_step)
    return (lambda *a: manual_flash(*a, step=step)), None


def kernel_us(fn, args, prep):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(A.warmup):
        if prep:
            prep()
        out = fn(*args)
        out.cpu()
    reps = []
    for _ in range(A.reps):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            if prep:
                prep()
            fn(*args).cpu()
        t = 0.0
        for ev in prof.key_averages():
            us = getattr(ev, "self_device_time_total", 0) or 0
            nm = ev.key or ""
            if us > 0 and "Memset" not in nm and "Memcpy" not in nm:
                t += us
        if t > 0:
            reps.append(t)
    return statistics.median(reps) if reps else float("nan")


def main():
    B, H, Lq, Lk, D = A.b, A.h, A.lq, A.lk, A.d
    live_kb = 2 * (min(A.bh, H) * min(A.bq, Lq) * min(A.bk, Lk)) * (2 if SPYRE else 4) / 1024
    print(f"MODE {A.mode}  device={A.device}  B{B} H{H} Lq{Lq} Lk{Lk} D{D}")
    if A.mode.startswith("manual"):
        print(f"  blocks: bh={A.bh} bq={A.bq} bk={A.bk}  -> "
              f"{(H + A.bh - 1)//A.bh * ((Lq + A.bq - 1)//A.bq) * ((Lk + A.bk - 1)//A.bk)} block steps, "
              f"live score block ~{live_kb:.0f} KB (this is what must fit in LX)")
    else:
        print(f"  hint: tiles H={A.h_tiles} Lq={A.lq_tiles} Lk={A.lk_tiles}  work_div={A.wd}")
    if SPYRE:
        print(f"  LX: frac_avail={os.environ.get('DXP_LX_FRAC_AVAIL', '0.2')} "
              f"(budget ~{2048 * (1 - float(os.environ.get('DXP_LX_FRAC_AVAIL', '0.2'))):.0f} KB/core), "
              f"all_ops={A.lx_all} boundary_clones={A.lx_boundary} solver={A.lx_solver or 'greedy'} "
              f"planning={'off' if A.no_lx else 'on'}")

    if A.plan_only:
        print(f"RESULT: PLAN  {A.mode}  live_block_kb={live_kb:.0f}")
        return 0

    torch.manual_seed(0)
    q = torch.rand(B, H, Lq, D, dtype=DT)
    k = torch.rand(B, H, Lk, D, dtype=DT)
    v = torch.rand(B, H, Lk, D, dtype=DT)
    mask = None if A.no_mask else torch.zeros(1, 1, Lq, Lk, dtype=DT)  # broadcast, as in the example
    ref = reference(q, k, v, mask) if not A.no_check else None
    args = tuple(None if t is None else t.to(DEV) for t in (q, k, v, mask))

    fn, prep = build(*args)
    out = fn(*args)
    if ref is not None:
        got = out.cpu().float()
        err = (got - ref.float()).abs().max().item()
        rel = err / max(1e-9, ref.float().abs().max().item())
        ok = rel < (2e-2 if DT is torch.float16 else 1e-4)
        print(f"  correctness: max|err|={err:.3e} rel={rel:.2e}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(f"RESULT: WRONG  {A.mode}  rel={rel:.2e}")
            return 3
    if not SPYRE:
        print(f"RESULT: OK  {A.mode}  (cpu: math verified, no timing)")
        return 0
    us = kernel_us(fn, args, prep)
    print(f"RESULT: OK  {A.mode}  kernel_us={us:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
