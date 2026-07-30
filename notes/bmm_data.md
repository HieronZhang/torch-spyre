# Batched-matmul (bmm) data  (from notes/sweep_records.json)

Source: `haoyang_logs/bmm_20260714_200003.log` (untiled), `haoyang_logs/coarse_tiling_20260714_190916.log` (tiled).
MACs = TRUE `B·M·N·K` from shape. `Aᵗ`,`Bᵗ` = TRUE operand MB from shape.
`%peak` = `MACs/(1140·cores)` ÷ kernel_us (measured compute utilization).
`splits` = `op_it_space_splits` d0..d3 (compiler work-division; d0=batch, never core-split for B≥2).
(`effBW`=io/kernel dropped: for matmul the kernel time is a composition of compute+HBM+psum,
so a single io/time "bandwidth" is not a meaningful quantity.)

## UNTILED (tiles=1) — this is the focus

### FULL bmm (per-batch weight `[B,K,N]`)

| B | M | K | N | MACs | splits | µs/B | kernel µs | pred µs | err% | %peak |
|--|--|--|--|--|--|--|--|--|--|--|
| 1 | 1024 | 2048 | 1024 | 2.1e+09 | 8x4x1     | 148  | 148  | 114  | -23 | 40% |
| 2 | 1024 | 2048 | 1024 | 4.3e+09 | 1x16x2x1  | 480  | 960  | 485  | -50 | 12% |
| 4 |  256 | 2048 | 1024 | 2.1e+09 | 1x8x2x2   | 113  | 453  | 250  | -45 | 13% |
| 4 |  512 | 2048 | 1024 | 4.3e+09 | 1x16x2x1  | 223  | 893  | 429  | -52 | 13% |
| 4 | 1024 |  512 | 1024 | 2.1e+09 | 1x16x2x1  | 222  | 886  | 244  | -72 | 7%  |
| 4 | 1024 | 1024 | 1024 | 4.3e+09 | 1x16x2x1  | 466  | 1864 | 427  | -77 | 6%  |
| 4 | 1024 | 2048 | 1024 | 8.6e+09 | 1x16x2x1  | 946  | 3785 | 798  | -79 | 6%  |
| 4 | 2048 | 2048 | 1024 | 1.7e+10 | 1x16x2x1  | 1470 | 5880 | 1536 | -74 | 8%  |
| 8 | 1024 |   64 | 1024 | 5.4e+08 | 1x16x2x1  | 38   | 300  | 154  | -48 | 5%  |
| 8 | 1024 | 2048 | 1024 | 1.7e+10 | 1x16x2x1  | 924  | 7394 | 1329 | -82 | 6%  |

(three duplicate B=4,1024×2048×1024 rows at 3798/3803/3810 µs omitted; run-to-run ±0.7%)

### 3d2d (shared 2-D weight `[K,N]`)

| B | M | K | N | MACs | splits | µs/B | kernel µs | pred µs | err% | %peak |
|--|--|--|--|--|--|--|--|--|--|--|
| 4 | 1024 | 2048 | 1024 | 8.6e+09 | 1x8x4x1 | 151 | 603  | 753  | +25 | 39% |
| 8 | 1024 | 2048 | 1024 | 1.7e+10 | 1x8x4x1 | 187 | 1498 | 1223 | -18 | 31% |

## TILED (coarse-tiling, for reference — not the current focus)

### FULL bmm, B=4 1024×2048×1024

| tiles | splits | io MB | kernel µs | pred µs | err% | %peak |
|--|--|--|--|--|--|--|--|
| 1  | 1x16x2x1 | 42  | 3785 | 798  | -79 | 6% |
| 2  | 1x32x1   | 109 | 4398 | 1136 | -74 | 5% |
| 4  | 1x32x1   | 176 | 4949 | 1692 | -66 | 5% |
| 8  | 1x32x1   | 310 | 6092 | 2930 | -52 | 4% |
| 16 | 1x32x1   | 579 | 8364 | 5469 | -35 | 3% |

### 3d2d, B=4 1024×2048×1024

| tiles | splits | io MB | kernel µs | pred µs | err% | %peak |
|--|--|--|--|--|--|--|--|
| 1  | 1x8x4x1 | 29  | 603  | 753  | +25 | 39% |
| 2  | 1x32x1  | 96  | 1172 | 1052 | -10 | 20% |
| 4  | 1x32x1  | 164 | 1870 | 1608 | -14 | 13% |
| 8  | 1x32x1  | 298 | 3299 | 2846 | -14 | 7%  |
| 16 | 1x32x1  | 566 | 6104 | 5385 | -12 | 4%  |

### nested B+K, B=4 1024×2048×1024

| tiles | splits | io MB | kernel µs | pred µs | err% | %peak |
|--|--|--|--|--|--|--|--|
| 1 | 1x16x2x1 | 42 | 3792 | 798   | -79 | 6% |
| 2 | 1x32x1   | 67 | 4897 | 5240  | +7  | 5% |
| 4 | 1x32x1   | 67 | 5645 | 8360  | +48 | 4% |
| 8 | 1x32x1   | 67 | 7145 | 13376 | +87 | 3% |

## Key observations (untiled full bmm, fixed shape 1024×2048×1024)

- **Per-batch cost is NOT constant** — it rises with B then saturates:
  148 (B=1, split 8×4) → 480 (B=2) → 946 (B=4) → 924 (B=8).
  So "predict one 2-D matmul × B" undershoots: each batch inside the bmm costs
  ~6× a standalone (148 → ~930 µs/B).
- **B=1 gets a different split (8×4) than B≥2 (16×2).** The 8×4 B=1 runs at 40% of
  peak; every 16×2 case sits at 5–8%. Whether the thin-M 16×2 split is the *cause*
  of the per-batch jump, or just correlated, is UNCONFIRMED — see the forced-split
  experiment below.
- 3d2d (shared weight) at the same shape holds 31–39% of peak (µs/B ≈ 150–190,
  ~flat in B) — i.e. it behaves like the B=1 full-bmm point.
