# Spyre Cost Model — Status

A high-level analytical model that predicts **relative device kernel time** from the
after-pre-scheduling LoopLevel IR. Calibrated against the golden AIU profiler
("Self SPYRE" per-kernel device time). Bytes are **device/stick-padded** (a row of
N fp16 rounds up to a multiple of the 64-elem stick); LX-resident intermediates ≈ free.

## Harness ops (what each `BENCH_OP` actually runs)

| group | ops | torch expression |
|---|---|---|
| pointwise | `neg` `gelu` `exp` `mul` `add` | `-x`, `gelu(x)`, `exp(x)`, `a*b`, `a+b` |
| pointwise n-ary | `add3` `add4` | `a+b+c`, `a+b+c+d` (2 / 3 chained adds) |
| reduction | `sumrow` `sumcol` `amax` | `sum(x,dim=1)`, `sum(x,dim=0)`, `amax(x)` |
| broadcast | `bcast` `bcastcol` `write` | `a[R,C]+b[1,C]`, `a[R,C]+b[R,1]`, `b[1,C]+c[R,1]` |
| transport | `transpose` `cat0` `cat1` | `x.transpose(0,1).contiguous()`, `cat([x,x],dim=0/1)` |
| matmul | `mm` `mmwd` | `a@b` (planner split / forced `WD_M×WD_N×WD_K` split) |
| coarse-tiling | `softmax_row_tiling` `matmul_row_tiling` `chain` | `softmax(x,dim=-1)`, `a@b`, `(a+b)*c` — tiled so an intermediate stays in LX |

## (1) Model — math form

Per kernel, with `R`,`W` = HBM bytes read / written:

```
T = compute + HBM/eff − γ·min(compute, HBM/eff)
```

| term | form | meaning |
|---|---|---|
| `HBM` (default) | `(R+W)/BW_peak + α·min(R,W)` | bandwidth + read↔write bus **turnaround** |
| `HBM` (access-pattern op) | `(R+W)/BW_eff` | per-op effective BW (`transpose`/`cat0`/`sumcol`) |
| `HBM` (matmul) | `R/BW_r + W/BW_w + α·min(R,W) + spill` | two-rate reads/writes + operand re-read |
| `compute` (matmul only) | `MACs / cores / peak` | `MACs=M·N·K`, cores = split product; else `compute=0` |
| **`eff`** (underfill) | `min(1, (rows_per_core / r_full)^0.35)`, `r_full≈16` | **coarse-tiling only**: a short per-core tile underfills the streaming pipeline → derates HBM. `=1` untiled or when the tile is tall enough |
| `γ·min(compute,HBM)` | `γ=0.46` | compute/HBM **overlap** (memory hides behind compute) |
| `spill` | `|A|·f(M/m) + |B|·f(N/n)`, `f(t)=min(1.7, 1.1·log₂(t/448))` | matmul per-core operand **re-read** past on-chip capacity |
| n-ary derate | `× (1 + 0.075·(n_ops−1))` | multi-pass pointwise chain (`add3/add4`, HBM intermediates) |

Constants: `BW_peak=150`, `α=0.00574 ns/B`, matmul `BW_r=143 / BW_w=156`,
`peak=1140 MAC/ns/core`. K is always kept whole (`WD_K=1`); K-split is never used.

> **Coarse-tiling caveat (§5):** a coarse-tiled op is ONE *fused* kernel (intermediates stay
> in LX). The model builds `R`,`W` by **summing per-op bytes** — correct for an untiled chain
> (each op is a separate kernel + HBM round-trip) but WRONG when fused: an input read by two
> fused ops (e.g. `softmax`'s `arg0` in `amax` *and* `sub`) is loaded from HBM **once** and
> reused from LX, yet the sum counts it twice. This is the main coarse-tiling error today.

## (2) Data — ~180 points from the profiling DB (`run_db_sweep.sh` + re-read sweep)

**Coverage (whole tested range), predicted vs measured error:**

| category (ops) | n | tested range | err range | RMS |
|---|---|---|---|---|
| pointwise (`neg` `gelu` `exp` `mul` `add` `copy`) | 25 | C 512–16384 | −3…+15% | 5% |
| pointwise n-ary (`add3` `add4`) | 6 | C 1024–16384 | −3…+2% | 2% |
| reduction (`sumrow` `sumcol` `amax` `mean` `sumall` `read`) | 12 | C 2048–8192 | −2…+6% | 3% |
| transport (`transpose` `cat0` `cat1` `transpose_outer`) | 13 | R×C 512–8192 | −15…+5% | 5% |
| **matmul, balanced split** | 44 | MNK 2e9–3.4e10 | −43…+29% | **8%** (≈8% on power-of-2 shapes) |
| broadcast (`bcast` `bcastcol` `write`) ⚠ | 12 | C 16384 | −62…+27% | 24% |
| coarse `softmax_row_tiling` ⚠ | 18 | 1–32 tiles | −34…+31% | ~20% |
| coarse `chain` ⚠ | 12 | 1–64 tiles | −22…+3% | ~11% |
| coarse `matmul_row_tiling` ⚠ | 9 | 1–16 tiles | −38…+7% | ~20% |

**Representative points:**

| op | size M×K×N / R×C | split | pred | meas | err |
|---|---|---|---|---|---|
| `neg` | 2048×16384 | — | 1280 | 1314 | −2.6% |
| `add4` | 2048×16384 | — | 5959 | 6113 | −2.5% |
| `sumrow` | 2048×8192 | — | 227 | 224 | +1.1% |
| `sumcol` | 2048×8192 | — | 297 | 298 | −0.3% |
| `transpose` | 2048×2048 | — | 145 | 145 | −0.1% |
| `cat0` | 2048×2048 | — | 419 | 401 | +4.7% |
| `mmwd` (compute-bound) | 2048×2048×2048 | 2×2×1 | 2081 | 2018 | +3.2% |
| `mmwd` (cores=32) | 2048×4096×2048 | 4×8×1 | 665 | 667 | −0.3% |
| `mmwd` (spill, big M) | 8192×2048×2048 | 4×8×1 | 1590 | 1586 | +0.2% |

⚠ = **open items**: `bcast`/`write` (broadcast operand faster than the V-curve / write-only
slower); tiny matmuls (fixed-overhead floor); extreme *forced* splits (not what a planner
picks); **coarse-tiling** ops not yet re-fit — their per-tile overhead terms are still
mis-calibrated, so error grows with tile count.

## (3) How the matmul parameters are isolated (order matters)

Each term is fit in a regime where it **dominates**, subtracting only terms already
validated (never model-minus-model on an unvalidated term):

1. **HBM (`BW_r`, `BW_w`)** — use compute-free matmuls: thin K (K≤64 → output ≫ operands,
   write-dominated) and thin M (M≤64 → tiny output, read-dominated). Compute <10%, so
   `kernel ≈ HBM`; fit the two rates from bytes. → 143 / 156.
2. **compute (`peak`) + overlap (`γ`)** — force **low core counts** (4–8) so compute is
   80–90 % of the kernel; subtract the byte-based HBM → `peak≈1140`. A single peak
   over-predicts cores=32, so add `γ` (memory overlaps compute) and fit both jointly on
   the compute-dominant runs. → peak=1140, γ=0.46 (RMS 1.7 %).
3. **spill (the split effect)** — on the corrected compute+HBM baseline, the residual on
   balanced cores=32 runs is operand re-read. Decouple with two sweeps: vary the matrix
   dim at a **fixed split** (isolates the per-core tile) and vary fanout at a **fixed small
   tile** (proved fanout is *not* a term). → re-read is a per-operand log-curve in M/m, N/n.
   (**K-split `WD_K>1` is excluded** — the planner always keeps K whole; we confirmed the
   old `(k−1)·out` "psum" ring term is ≈ 0, so it is simply dropped.)

Result: **≈8–12 % RMS across the balanced matmul range** (cores 4→32, MNK 2e9→3.4e10;
≈8 % on power-of-2 shapes, the −40 % tail is non-power-of-2 N stick-padding, unmodeled).

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
| `transpose_outer` | 3-D outer swap, stick kept | 101 | ≈ same |
| `cat0` | concat on the **stick** dim | **63** | −40 % (fine sub-stick scatter) |

So a plain copy (`neg`) and most transports share the ~105–116 balanced rate; the two
outliers (`transpose` faster, `cat0` slower) get a per-op `BW_eff` that the extractor
reads straight from the IR (stick-var coefficient in the load index / device layout).

## (5) Coarse tiling — a fused kernel, NOT a sum of kernels (open rework)

`softmax_row_tiling` (`softmax(x,dim=-1)`), `chain` (`(a+b)*c`), `matmul_row_tiling` (`a@b`)
tile one dimension so a loop runs all the ops **fused into ONE kernel**, keeping the
intermediates in LX. The model handles the untiled case correctly (each op is a separate
kernel → sum the per-op HBM), but the fused case has a **structural error**:

- **Fatal error — double-counted reused inputs.** Summing per-op `R`/`W` counts an input
  read by several fused ops once *per op*. `softmax` reads `arg0` in both `amax` and `sub`,
  so the model charges `2×arg0` of HBM — but the fused kernel loads each tile **once** and
  `sub` reads it back **from LX**. The correct fused HBM is *distinct external inputs once +
  outputs once*; internal reuse and intermediates are LX (free), with compute pipelined
  behind that I/O. This is not a tunable "effective R" — it is a fixed 1× read that only
  breaks when a tile exceeds LX.
- **Tile-count trend, explained physically** (measured, not fit):

  | op | kernel vs #tiles | why |
  |---|---|---|
  | `softmax_row_tiling` | **decreases** | tile fits LX → `arg0`'s 2nd read is LX not HBM (→1× read); at few/big tiles the tile exceeds LX (~33–50 MB knee: `[16384,4096] t=2` = 67 MB tile spills, runs like untiled) so it re-reads |
  | `chain` | flat then **cliffs** | pure **underfill** — proven by the tall `[16384,512]` staying flat (rows/core ≥ `r_full`≈16) while wide `[2048,4096]` cliffs; the `eff` exponent (0.35) is too weak |
  | `matmul_row_tiling` | **U-shape / grows** | same underfill, but on the per-tile M — `pt_eff` must be keyed on `M/tiles`, not the whole M |

- **Fixes (in progress):** (a) count each distinct external HBM input once for a fused
  kernel + add an LX-capacity spill (the ~33–50 MB knee) when a tile overflows; (b) steepen
  the `eff` underfill exponent (fixes `chain`); (c) key `matmul_row_tiling`'s `pt_eff` /
  underfill on the coarse-tile `M/tiles`. Until then the coarse rows in §2 carry ~11–20 % RMS.
