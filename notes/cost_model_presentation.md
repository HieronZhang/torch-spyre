# Spyre Cost Model — Status

A high-level analytical model that predicts **relative device kernel time** from the
after-pre-scheduling LoopLevel IR. Calibrated against the golden AIU profiler
("Self SPYRE" per-kernel device time). Bytes are **device/stick-padded** (a row of
N fp16 rounds up to a multiple of the 64-elem stick); LX-resident intermediates ≈ free.

## Harness ops (what each `BENCH_OP` actually runs)

| group | ops | torch expression |
|---|---|---|
| pointwise | `neg` `gelu` `exp` `mul` `add` `copy` | `-x`, `gelu(x)`, `exp(x)`, `a*b`, `a+b`, `x+1.0` (`copy` is a broadcast op) |
| pointwise n-ary | `add3` `add4` | `a+b+c`, `a+b+c+d` (2 / 3 chained adds) |
| reduction | `sumrow` `sumcol` `amax` `mean` `sumall` `read` | `sum(x,dim=1)`, `sum(x,dim=0)`, `amax(x)`, `mean(x,dim=1)`, `sum(x)`, `x` (pure read) |
| broadcast | `bcast` `mulbcast` `bcastcol` `write` | `a[R,C]+b[1,C]`, `a[R,C]*b[1,C]`, `a[R,C]+b[R,1]`, `b[1,C]+c[R,1]` |
| transport | `transpose` `transpose_outer` `cat0` `cat1` | `x.transpose(0,1).contiguous()`, 3-D outer swap, `cat([x,x],dim=0/1)` |
| matmul | `mm` `mmwd` | `a@b` (planner split / forced `WD_M×WD_N×WD_K` split) |
| coarse-tiling | `softmax_row_tiling` `matmul_row_tiling` | `softmax(x,dim=-1)`, `a@b` — tiled so an intermediate stays in LX |

## (1) Model — math form

Per kernel, with `R`,`W` = HBM bytes read / written:

```
T = compute + HBM/eff − γ·min(compute, HBM/eff)
```

| term | form | meaning |
|---|---|---|
| `HBM` (default) | `(R+W)/BW_peak + α·min(R,W)` | bandwidth + read↔write bus **turnaround** |
| `HBM` (row-reduction) | `(R+W)/BW_red(ROWS)` | read-bound; rate `BW_red = min(150, 114+61·e^(−ROWS/3700))` falls with ROWS (op-independent) |
| `HBM` (access-pattern op) | `(R+W)/BW_eff` | per-op effective BW (`transpose`/`cat0`/`sumcol`; also broadcast-operand ops `copy`/`bcast`/`bcastcol`/`mulbcast`) |
| `HBM` (matmul) | `(R+W)/BW_peak + α·min(R,W) + spill` | single-rate (= copy peak) + operand re-read |
| `compute` (matmul only) | `MACs / cores / (peak · pt_eff)` | `MACs=M·N·K`, cores = split product; else `compute=0`. `pt_eff` = systolic-array fill |
| **`eff`** (coarse memory underfill) | `min(0.95, (rpc/13)^0.68)`, `rpc = ROWS/(cores·tiles)` | **coarse-tiling**, memory-bound (softmax): a short per-core tile underfills the streaming pipeline → derates HBM. Keys on ROWS, not tile bytes. `=1` untiled |
| `pt_eff` (matmul fill) | standalone: `min(1,(rows/64)^0.35)`; **tiled**: `min(1,(rows/72)^0.85)` | systolic-array fill (rows = per-core rows). A coarse-**tiled** matmul (`matmul_row_tiling`) underfills far steeper per tile than one big matmul |
| `γ·min(compute,HBM)` | `γ=0.46` | compute/HBM **overlap** (memory hides behind compute) |
| `spill` | `|A|·f(M/m) + |B|·f(N/n)`, `f(t)=min(1.7, 1.1·log₂(t/448))` | matmul per-core operand **re-read** past on-chip capacity |
| n-ary derate | `× (1 + 0.075·(n_ops−1))` | multi-pass pointwise chain (`add3/add4`, HBM intermediates) |

Constants: `BW_peak=150`, `α=0.00574 ns/B`, matmul HBM `= single 150` (the old two-rate
`143/156` was retired — `156>150` is unphysical, a compute-free-fit artifact),
`peak=1140 MAC/ns/core`. Per-op effective BW: `restickify=116`, `stick_scatter` shape-dependent
(cat0: `252−4·log2R−12.3·log2C` clamped [45,150], falls with row width C),
`reduce_outer=113`, `broadcast=118` (an op streaming a full input + a broadcast operand,
e.g. `copy=x+1.0`, runs above the plain 1R:1W rate). `write` (both operands broadcast →
outer-product) adds an empirical extra HBM term `2.15e-7·ROWS^1.6·COLS^2.2` (black-box).
Standalone row-reductions (`sum`/`amax`/`mean`/`sumall`/`read` over the last axis) read at a
ROWS-derated rate `min(150, 114+61·exp(−ROWS/3700))` (op-independent; `sumcol` keeps
`reduce_outer=113`). K is always kept whole (`WD_K=1`).

> **Coarse-tiling (§5):** a coarse-tiled op is ONE *fused* kernel (intermediates stay in LX).
> `R`,`W` therefore count each distinct **external** input once + outputs once (`_fused_hbm_bytes`):
> `softmax`'s `arg0`, read by `amax` *and* `sub`, is loaded from HBM **once** (2nd read served
> on-chip) — the old per-op sum double-counted it (~+25% at the floor). Fixed 2026-07-09.

## (2) Data — ~180 points from the profiling DB (`run_db_sweep.sh` + re-read sweep)

**Coverage (whole tested range), predicted vs measured error:**

| category (ops) | n | tested range | err range | RMS |
|---|---|---|---|---|
| pointwise (`neg` `gelu` `exp` `mul` `add`) | 43 | C 512–16384 | −6…+6% | 2.2% |
| reduction (`sumrow` `sumcol` `amax` `mean` `sumall` `read`) | 58 | ROWS 2048–16384 | −2…+6% | 2.6% (ROWS-derated) |
| transport (`transpose` `cat0` `cat1` `transpose_outer`) | 42 | R×C 512–8192 | −22…+22% | 8.5% (cat0 shape-modeled; t_outer flagged) |
| **matmul, planner-realistic** | 34 | MNK 2e9–3.4e10 | −17…+26% | **6.9%** (K whole, fanout ≤8, pow-2 N) |
| broadcast (`copy` `bcast` `bcastcol` `mulbcast` `write`) | 51 | R×C to 16384 | −24…+14% | 7.7% |
| coarse `softmax_row_tiling` (non-spill) | 44 | rpc 2–512 | −14…+11% | **7.2%** |
| coarse `softmax_row_tiling` (LX-spill) ⚠ | 7 | rpc ≥160 | −40…−18% | 24% (spilled-traffic rate, deferred) |
| coarse `matmul_row_tiling` ⚠ | 9 | 1–16 tiles | −38…+9% | ~20% (deferred: `pt_eff` on M/tiles) |

**Representative points:**

| op | size M×K×N / R×C | split | pred | meas | err |
|---|---|---|---|---|---|
| `neg` | 2048×16384 | — | 1280 | 1314 | −2.6% |
| `add4` | 2048×16384 | — | 5959 | 6113 | −2.5% |
| `sumrow` | 2048×8192 | — | 227 | 224 | +1.1% |
| `sumcol` | 2048×8192 | — | 297 | 298 | −0.3% |
| `transpose` | 2048×2048 | — | 145 | 145 | −0.1% |
| `cat0` | 2048×2048 | — | 419 | 401 | +4.7% |
| `mmwd` (compute-bound) | 2048×2048×2048 | 2×2×1 | 2080 | 2014 | +3.3% |
| `mmwd` (cores=32) | 2048×4096×2048 | 4×8×1 | 661 | 667 | −1.0% |
| `mmwd` (spill, big M) | 8192×2048×2048 | 4×8×1 | 1585 | 1594 | −0.6% |

⚠ = **open items**: `bcast`/`write` (broadcast operand faster than the V-curve / write-only
slower); tiny matmuls (fixed-overhead floor); extreme *forced* splits (not what a planner
picks); **coarse-tiling** ops not yet re-fit — their per-tile overhead terms are still
mis-calibrated, so error grows with tile count.

## (3) How the matmul parameters are isolated (order matters)

Each term is fit in a regime where it **dominates**, subtracting only terms already
validated (never model-minus-model on an unvalidated term):

1. **HBM (single rate)** — use compute-free matmuls: thin K (K≤64 → output ≫ operands,
   write-dominated) and thin M (M≤64 → tiny output, read-dominated). Compute <10%, so
   `kernel ≈ HBM`. The dominant-operand rate is ~118–148 (write corners) / ~123–136 (read
   corners) — overlapping, both **below** the 150 copy peak → a **single 150** rate (an
   earlier two-rate 143/156 fit was retired; `156>150` was an unphysical overlap artifact).
2. **compute (`peak`) + overlap (`γ`)** — force **low core counts** (4–8) so compute is
   80–90 % of the kernel; subtract the byte-based HBM → `peak≈1140`. A single peak
   over-predicts cores=32, so add `γ` (memory overlaps compute) and fit both jointly on
   the compute-dominant runs. → peak=1140, γ=0.46 (RMS 1.7 %).
3. **spill (the split effect)** — on the corrected compute+HBM baseline, the residual on
   balanced cores=32 runs is operand re-read. Decouple with two sweeps: vary the matrix
   dim at a **fixed split** (isolates the per-core tile) and vary fanout at a **fixed small
   tile** (proved fanout is *not* a term). → re-read is a per-operand log-curve in M/m, N/n.
   (**K-split `WD_K>1` is excluded** — the planner always keeps K whole. The old `(k−1)·out`
   "psum" ring term is now **gated off matmul**: forcing `WD_K>1` made it explode (+489%), and
   since K is never split it contributes nothing in practice.)

Result: **≈6.9 % RMS on the planner-realistic envelope** (K whole, fanout ≤8, non-tiny,
pow-2 N; cores 4→32, MNK 2e9→3.4e10). Out-of-regime rows are flagged, not fit: **forced
K-splits −41 %** (unmodeled cross-core combine; the planner avoids K-splits), skewed splits
(fanout >8) −24 %, tiny operands (fixed-overhead floor), non-power-of-2 N ≈ −16 %
(stick-padding). See report §12 for the full regime table.

## (4) Transport ops are just copies

`transpose` / `cat0` / `cat1` lower in the LoopLevel IR to a **`Pointwise` `clone`** — a
byte-for-byte copy (`R = W = data`), identical to `neg` except the load index may be
reordered. The model therefore treats them exactly like a copy: `T=(R+W)/BW_eff`. The
**only** difference is the effective BW, set by how the copy touches the 64-elem stick
(measured as io_bytes / kernel_time):

| op | access | effBW (GB/s) | vs `neg` |
|---|---|---|---|
| `neg` | contiguous copy | 106 | baseline |
| `transpose` | swaps stick ↔ row | **116** | +10 % (restickify pays less turnaround) |
| `cat1` | concat on the outer dim | 108 | ≈ same |
| `transpose_outer` | 3-D outer swap, stick kept | 106→85 (falls with C) | flagged (no IR tag) |
| `cat0` | concat on the **stick** dim | 110→49 (falls with C) | shape-modeled |

So a plain copy (`neg`) and most transports share the ~105–116 balanced rate; the outliers get a
per-op `BW_eff` the extractor reads from the IR: `transpose` a fixed 116 (faster), `cat0` a
shape-dependent rate `252−4·log2R−12.3·log2C` (falls with row width C, R²0.93 over 10 shapes).
`transpose_outer` shows the SAME C-falloff (−22% at wide C) but carries no IR pattern tag, so it
stays on the default copy model, flagged.

## (5) Coarse tiling — a fused kernel, NOT a sum of kernels

`softmax_row_tiling` (`softmax(x,dim=-1)`) and `matmul_row_tiling` (`a@b`) tile one dimension
so a loop runs all the ops **fused into ONE kernel**, intermediates in LX. Two mechanisms,
isolated by the `softmax_terms` grid + a cross-COLS control (COLS 2048 vs 4096 at matched
per-core tile) + an adversarial challenge; `rpc = ROWS/(cores·tiles)` = per-core rows per tile:

- **Fused HBM — external input counted once (fixed 2026-07-09).** `softmax` reads `arg0` in
  both `amax` and `sub`; the fused kernel loads it from HBM **once** and serves the 2nd read
  on-chip. Counting it per-op over-charged the floor ~25%. Confirmed physically: at the
  underfill-free floor softmax runs at ~100 GB/s = the balanced-copy rate (1 read + 1 write).
- **Underfill keys on ROWS (rpc), not tile bytes, and not the tile count `L`.** At matched
  `rpc`, doubling COLS (2× tile bytes) leaves per-byte cost unchanged (±4%); and four `T=4..32`
  points at `rpc=16` cost the same (so `L` is not the driver). `eff` = `min(0.95,(rpc/13)^0.68)`:
  plateau ~0.95 at rpc 16–32, cliff below (rpc4≈0.45, rpc2≈0.28). Softmax now **RMS 7.3%**
  (floor ±2%; residual: rpc≤8 +8–10%, rpc≥64 −7…−14% — a mild rows-driven rise the cap omits).
- **LX-spill is auto-captured, not a modeled knee.** When a per-core tile overflows LX (~1–2
  MB/core) the compiler spills intermediates to HBM and the **IR already reflects it** (the
  extractor counts the extra bytes). Remaining gap: spilled traffic runs ~34% slower than the
  byte model (1 data point) — a rate effect, deferred to a finer knee sweep.

`chain` was dropped (per user). `matmul_row_tiling` still open — same underfill but on the
per-tile `M`; needs `pt_eff` keyed on `M/tiles`.
