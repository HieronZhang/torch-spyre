# Spyre Cost Model — Current Coverage & Matmul Plan

A short map of what the analytical cost model covers today and how matmul is
being folded in. For the full design, calibration data, and verification
ladder, see [cost_model_design.md](cost_model_design.md); for the in-tree
work-division Pass-2 matmul cost model (a *relative* split-picker, separate from
this one), see
[work_division_planning.md](../docs/source/compiler/work_division_planning.md).

- **Goal:** predict the *relative* per-kernel device latency from the
  after-pre-scheduling LoopLevel IR — not a simulator.
- **Golden measurement:** the `torch.profiler` "Self SPYRE" (`sdsc_fused_*`)
  kernel device time. The separate "Memset (Device)" event is setup overhead,
  not kernel time.
- **Code:** [`cost_model.py`](../torch_spyre/_inductor/cost_model.py) (the pure
  model + params) and
  [`dump_cost_model.py`](../torch_spyre/_inductor/dump_cost_model.py) (feature
  extraction; `SPYRE_DUMP_COST=1` prints the per-tensor I/O + the prediction).

## Scope

| Op class | Status | Extra term beyond bandwidth |
|---|---|---|
| Pointwise (unary / binary / n-ary fused) | ✅ validated | — |
| Reductions (sum / amax / mean, dim-0 / dim-1) | ✅ validated | ring `combine` |
| Coarse-tiled reduction (tile the reduced dim) | ✅ validated | `c_loop · L` |
| Coarse-tiled pointwise chain (intermediate in LX) | ✅ (new) | `eff_underfill` + `c_loop · L` |
| Matmul / bmm | 🚧 in progress | additive `compute` term (form validated) |

## The model today

```text
T = fill + [ (R + W) / BW_PEAK + α · min(R, W) ] / eff_underfill
        + combine + c_loop · L
```

`R` / `W` are the kernel's total HBM bytes **read** / **written** (each tensor
counted once at its stick-padded device size; LX-resident traffic is ~free;
broadcast operands are loaded once, not per core).

1. **Bandwidth (turnaround)** — `(R+W)/BW_PEAK + α·min(R,W)`. A single peak HBM
   bandwidth (`BW_PEAK ≈ 150 GB/s`, the one-directional rate) minus a read/write
   **turnaround** penalty on the overlap `min(R,W)` (`α ≈ 0.0057 ns/B`). This
   reproduces the measured V-shaped effective BW: ~150 read-only, ~105 at a
   balanced 1R+1W, back up for write-only. Covers pointwise and (because a
   reduction has a tiny write, so `min(R,W)≈0` → read-only rate) reductions.
2. **`eff_underfill`** (≤1) — derates the bandwidth term for **output-dim
   (pointwise) coarse-tiling** that shrinks each core's per-tile height. A tile
   shorter than `pass_rows · target_passes` underfills the streaming pipeline:
   `eff = min(1, (rows_per_core / r_full)^exp)`. Same FORM as the matmul
   `pt_eff`; the 8-row pass is a shared hardware constant. This is a **per-tile-
   SIZE** effect, **distinct from** (and added on top of) the per-iteration
   `c_loop·L` below. The chain K-sweep is flat to ~16 rows/core then cliffs (8
   rows/core ≈ +34%) — a flat-then-cliff shape a *linear* `c_loop·L` cannot
   produce, so underfill is the **dominant** pointwise-tiling term while
   `c_loop·L` stays the small loop-dispatch cost. (`r_full ≈ 16`, `exp ≈ 0.5` are
   PROVISIONAL; in this sweep `L` and `rows/core` are confounded — the
   untiled-small-ROWS confirm runs isolate the underfill.)
3. **`combine`** — cross-core ring reduction, `(k−1)·out_elems·psum_per_elem`,
   when the reduced axis is split across `k` cores (`psum_per_elem ≈ 0.14 ns`;
   the same constant the matmul PSUM term uses).
4. **`c_loop · L`** — per-**iteration** coarse-tiling loop-dispatch overhead, for
   **any** tiled loop (`c_loop ≈ 860 ns/tile`, `L` = trip count). A *different*
   mechanism from the underfill derate (fixed cost per iteration vs throughput
   loss per short tile). It dominates the tiling cost for a standalone reduction
   (whose tiny output doesn't underfill) and is the small term for a pointwise
   chain. (860 ns is calibrated from the reduction K-sweep; the pointwise value
   is confounded with underfill until the confirm runs separate them.)

**Parameters** (`CostParams`): `BW_PEAK=150`, `α=0.00574`, `psum_per_elem=0.14`,
`c_loop=860`, underfill `pass_rows=8 / target_passes=2 / exp=0.5`.

**Accuracy** (B–F profiler sweep): ~2% on core pointwise + reductions, ~7%
overall; the pointwise tiling cliff is now ~2% at 8 rows/core (was −30% with the
constant `c_loop`). Known per-category biases (so cross-kind comparisons can be
off ~15–20%): broadcast pointwise ~17% fast, write-only ~16%, fan-in add3/add4
~8%, dim-0 `sumcol` ~19% (access-pattern, not ring-combine).

## Matmul plan

Matmul is a `Reduction` with `reduction_type = batchmatmul`, a 3-D iteration
space `{M, K, N}` (output dims `M`,`N`; reduction dim `K`), split across cores
as `(b, m, n, k)`. Unlike everything above it is **compute-bound**, so it gets
one extra **additive** term:

```text
T_matmul = compute + H_turnaround + psum

compute       = MACs / cores / (mac_peak · pt_eff)      # MACs = M·N·K
pt_eff        = underfill_eff(M/m, target_passes = 8)   # PT-array fill derate
H_turnaround  = (R+W)/BW_PEAK + α·min(R,W)              # the bandwidth term above
psum          = (k−1) · M·N · psum_per_elem             # K-split ring (k>1 only)
```

**Why this exact form — validated on the `mm` K-sweep** (`M=N=2048`,
`K ∈ {512…8192}`, every run split `M:4 N:8 K:1`, so `k=1` → no PSUM, clean
compute + HBM):

| K | measured µs | `compute_datasheet + H_turnaround` | err |
|---|---|---|---|
| 512 | 163.4 | 151.7 | −7.2% |
| 1024 | 244.1 | 247.4 | +1.3% |
| 2048 | 389.0 | 390.7 | +0.4% |
| 4096 | 667.5 | 677.3 | +1.5% |
| 8192 | 1240.7 | 1250.5 | +0.8% |

Three findings, robust across a 16× range in `K` (not a single-point fit):

- **Additive — no compute/HBM overlap.** The `max(compute, HBM)` (perfect
  double-buffering) model is off by 34–48%; the sum fits to ~1%.
- **The datasheet MAC peak is correct** (≈49 TMAC/s effective, `1536 MAC/ns/core`).
  The additive fit uses it untouched. (An earlier "≈45% efficiency" guess was
  circular and is falsified — a fixed efficiency would swing 27→56% across this
  sweep; it doesn't.)
- **The only broken constant in the in-tree Pass-2 model was its bandwidth**
  (`204.8 GB/s`). Swapping it for this turnaround BW takes matmul from −24…−56%
  to ~1%. `pt_eff` and the additive structure carry over unchanged.

**`pt_eff` reuses the `eff_underfill` derate** with `target_passes = 8` (matmul
saturates ~64 rows/core vs pointwise ~16) — one shared pipeline-fill mechanism,
two op-dependent saturation points.

**Status / open:**

- *Implemented:* the `compute` term in `cost_model.py` (params
  `mac_peak_per_core_ns=1536`, `underfill_target_passes_matmul=8`) and the
  extraction of `MACs` / `m` / `k` from `op_it_space_splits` in
  `dump_cost_model.py`, so `mm` runs now get a real prediction (no longer just the
  bandwidth-only estimate). Reproduces the K-sweep to ~1% (−7% at K=512); a forced
  K-split correctly adds the PSUM term. Still needs an on-device re-run to confirm
  the extraction path (`m`/`k` from `op_it_space_splits`) on hardware.
- *Open — small-kernel overhead:* `K=512` is +7% above additive (a fixed ~10 µs
  startup is a larger fraction of a small kernel); to be pinned later.
- *Open — cohort penalty:* every K-sweep run had `max(m,n)/8 = 1`, so the
  operand-broadcast contention term never engaged. A split with `m` or `n` > 8
  is needed to test it.
