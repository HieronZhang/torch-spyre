# The coarse-tiling cost model — current state

*Generated from the live model by `python3 notes/coarse_tables.py`. Nothing here is
hand-typed; re-run after any coefficient change.*

## What a coarse-tiled kernel is

Coarse tiling turns one big operation into a **sequential loop over tiles**, so the
compiler can keep a working set on-chip. The IR records this as a `CoarseTileInfo`
stamped on every op in the loop nest:

```text
loop_group_id = (0,)        or (0, 0) when nested
loop_count    = [T]         or [T_outer, T_inner]
loop_tiled_dims           = [[0]]      # which OUTPUT dims each level tiles
loop_tiled_reduction_dims = [[]]       # which REDUCTION dims each level tiles
```

Three shapes occur in the data: tiling an **output** dim (`matmul_row_tiling`, tiles M),
tiling the **reduction** dim (`matmul_k_tiling`, tiles K), and **nested** (`mm_nested_m_k`
= M outer, K inner). `softmax_row_tiling` is a fused multi-op region — five
ComputedBuffers sharing one `loop_group_id`.

## The model

```text
T = compute + mem_t - gamma(rho) * min(compute, mem_t) + split

  compute = MACs / cores / (mac_peak * pt_eff)
  mem_t   = [ (R_base + W)/BW + alpha*min(R_base, W) + spill ] / eff / s_lx
            + reread_scale * REREAD / BW
  rho     = min(compute, mem_t) / max(compute, mem_t)
```

**The load-bearing piece is `R`, and it comes from the loop structure.** Each argument is
transferred a number of times set by its own index expression:

```text
factor(arg) = PRODUCT over nesting levels L of
                ( loop_count[L]  if the arg's index contains NO tiled symbol of level L
                  else 1 )
```

An argument whose address does not depend on a level's tiled symbol is re-entered at the
same address every iteration of that level, so it moves again. This is derived from the
IR, not fitted. For `matmul_row_tiling` at t=4 the IR reads

```text
tmp0 = ops.load(arg0_1, r0_0 + 2048*i0)   # A -- contains the tiled symbol i0 -> factor 1
tmp1 = ops.load(arg1_1, i1   + 2048*r0_0) # B -- does NOT contain i0        -> factor 4
```

giving out/A/B = 1/1/4. K-tiling gives 4/1/1 (the accumulator repeats, both operands
advance); the nested op gives 4/1/2 — its OUTPUT advances at the M level and repeats at
the K level, which a single per-op scalar cannot express.

`REREAD` is the excess over the first pass (`elems * (factor - 1) * dtype`) and is charged
**outside** the `eff`/`s_lx` derates: those model a short per-tile stream, whereas a
re-read is one large contiguous pass over a whole operand.

## Honest status of each term

| term | basis |
|---|---|
| per-arg `loop_factor` | **derived** from the IR; unit-verified 13/13 |
| re-read charged at all | **derived** (B is provably not LX-resident: the allocator declines to pin an input it sees used once) |
| `reread_scale`, `gamma(rho)` | **EMPIRICAL, fitted jointly** — they are confounded at r ~ -0.90 and cannot be separated by this data |
| matmul spill cap 2 MB | existing term, matmul-specific capacity |

An adversarial review refuted the original *mechanistic* framing of `gamma(rho)`:
26 of 125 affected rows individually require a hidden fraction > 1.0 (max 1.29), which no
overlap can do; a free re-fit puts the endpoint at 1.27 (bootstrap P(>1.0) = 1.00). So the
term absorbs a non-overlap over-charge, `rho` explains at most half the effect, and the
near-balanced band (rho >= 0.9) is a known **regression** (RMS 9.6 -> 10.7). It is kept
only because it is the best correction found (leave-one-shape-out CV 8.5 vs 9.9 for a pure
re-read rescale). The deciding experiment is a sweep at fixed re-read share varying rho,
and vice versa.

### Coefficients that act on a coarse-tiled kernel

| coefficient | value | role |
|---|---:|---|
| `BW (read/write)` | 150 GB/s | HBM streaming rate |
| `alpha turnaround` | 0.00574 ns/B | read/write bus turnaround |
| `MAC peak` | 1140 MAC/ns/core | systolic-array rate |
| `gamma (balanced)` | 0.46 | compute/memory overlap at rho -> 1 |
| `gamma (unbalanced)` | 1.0 | overlap at rho -> 0 (EMPIRICAL, see caveats) |
| `gamma exponent` | 0.6 | shape of gamma(rho) |
| `re-read scale` | 0.85 | fraction of a full HBM pass per repeat |
| `matmul spill cap` | 2 MB | per-core LX before the tile spills |
| `matmul spill exp` | 0.15 | spilled-traffic derate |
| `fused elem rate` | 1.51 | per-core element throughput floor (softmax) |
| `coarse underfill` | r_full=13, exp=0.68, cap=0.95 | short-tile pipeline fill (softmax-calibrated) |

### Accuracy

| op | n | RMS % | mean % | worst % | >20 % |
|---|---:|---:|---:|---:|---:|
| `matmul_row_tiling` | 143 | 7.5 | -1.9 | +19.1 | 0 |
| `softmax_row_tiling` | 64 | 7.2 | -0.3 | +21.7 | 1 |
| `softmax_noexp_row_tiling` | 3 | 5.3 | +5.3 | +5.9 | 0 |
| `matmul_k_tiling` | 1 | 7.7 | -7.7 | -7.7 | 0 |
| `mm_nested_m_k` | 1 | 7.7 | -7.7 | -7.7 | 0 |
| **all** | **212** | **7.4** | **-1.4** | **+21.7** | **1** |

| \|err\| band | count |
|---|---:|
| 0-5 % | 98 |
| 5-10 % | 76 |
| 10-15 % | 28 |
| 15-20 % | 9 |
| 20-&infin; % | 1 |

### Every coarse-tiling data point in the target band

| op | M | K | N | t | rows/core | measured us | predicted us | err % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `matmul_k_tiling` | 4096 | 2048 | 2048 | 1 | 512 | 846.2 | 781.2 | -7.7 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 1 | 256 | 389.6 | 393.4 | +1.0 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 1 | 256 | 387.9 | 393.4 | +1.4 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 1 | 256 | 387.4 | 393.4 | +1.5 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 1 | 256 | 390.2 | 393.4 | +0.8 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 1 | 256 | 387.2 | 393.4 | +1.6 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 1 | 256 | 386.2 | 393.4 | +1.9 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 2 | 128 | 340.8 | 362.4 | +6.3 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 2 | 128 | 341.3 | 362.4 | +6.2 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 2 | 128 | 335.0 | 362.4 | +8.2 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 2 | 128 | 339.2 | 362.4 | +6.8 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 4 | 64 | 436.9 | 427.8 | -2.1 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 4 | 64 | 436.9 | 427.8 | -2.1 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 4 | 64 | 436.1 | 427.8 | -1.9 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 4 | 64 | 440.5 | 427.8 | -2.9 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 4 | 64 | 435.7 | 427.8 | -1.8 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 4 | 64 | 434.2 | 427.8 | -1.5 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 4 | 64 | 434.9 | 427.8 | -1.6 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 8 | 32 | 658.0 | 595.5 | -9.5 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 8 | 32 | 657.3 | 595.5 | -9.4 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 8 | 32 | 656.6 | 595.5 | -9.3 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 8 | 32 | 652.1 | 595.5 | -8.7 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 8 | 32 | 658.1 | 595.5 | -9.5 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 8 | 32 | 653.4 | 595.5 | -8.9 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 8 | 32 | 656.4 | 595.5 | -9.3 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 16 | 32 | 1002.0 | 960.5 | -4.1 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 16 | 32 | 999.6 | 960.5 | -3.9 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 16 | 32 | 998.9 | 960.5 | -3.8 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 1 | 512 | 769.9 | 781.2 | +1.5 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 1 | 512 | 770.9 | 781.2 | +1.3 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 1 | 512 | 773.5 | 781.2 | +1.0 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 2 | 256 | 752.4 | 758.0 | +0.8 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 4 | 128 | 728.1 | 809.7 | +11.2 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 4 | 128 | 724.2 | 809.7 | +11.8 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 4 | 128 | 728.8 | 809.7 | +11.1 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 4 | 128 | 727.7 | 809.7 | +11.3 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 8 | 64 | 1127.4 | 1136.7 | +0.8 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 8 | 64 | 1122.5 | 1136.7 | +1.3 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 8 | 64 | 1131.3 | 1136.7 | +0.5 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 8 | 64 | 1129.1 | 1136.7 | +0.7 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 16 | 32 | 1946.1 | 1863.5 | -4.2 |
| `matmul_row_tiling` | 2048 | 4096 | 2048 | 1 | 256 | 671.5 | 702.3 | +4.6 |
| `matmul_row_tiling` | 2048 | 4096 | 2048 | 2 | 128 | 597.3 | 662.3 | +10.9 |
| `matmul_row_tiling` | 2048 | 4096 | 2048 | 4 | 64 | 774.6 | 773.1 | -0.2 |
| `matmul_row_tiling` | 2048 | 4096 | 2048 | 8 | 32 | 1217.8 | 1090.3 | -10.5 |
| `matmul_row_tiling` | 4096 | 2048 | 512 | 1 | 512 | 206.0 | 241.4 | +17.2 |
| `matmul_row_tiling` | 4096 | 2048 | 512 | 4 | 128 | 206.4 | 245.9 | +19.1 |
| `matmul_row_tiling` | 4096 | 2048 | 512 | 8 | 64 | 276.2 | 288.9 | +4.6 |
| `matmul_row_tiling` | 4096 | 2048 | 512 | 16 | 32 | 411.4 | 378.7 | -7.9 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 1 | 512 | 419.3 | 434.0 | +3.5 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 1 | 512 | 421.6 | 434.0 | +2.9 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 2 | 256 | 388.2 | 365.2 | -5.9 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 4 | 128 | 435.5 | 395.7 | -9.1 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 4 | 128 | 435.3 | 395.7 | -9.1 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 8 | 64 | 441.8 | 472.2 | +6.9 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 8 | 64 | 441.5 | 472.2 | +7.0 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 8 | 64 | 441.4 | 472.2 | +7.0 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 16 | 32 | 656.2 | 645.3 | -1.7 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 16 | 32 | 662.4 | 645.3 | -2.6 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 32 | 32 | 1072.8 | 1012.6 | -5.6 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 1 | 512 | 844.8 | 781.2 | -7.5 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 1 | 512 | 836.4 | 781.2 | -6.6 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 1 | 512 | 837.6 | 781.2 | -6.7 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 1 | 512 | 845.5 | 781.2 | -7.6 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 1 | 512 | 841.3 | 781.2 | -7.1 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 1 | 512 | 835.0 | 781.2 | -6.4 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 1 | 512 | 844.4 | 781.2 | -7.5 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 2 | 256 | 783.5 | 707.9 | -9.7 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 2 | 256 | 776.4 | 707.9 | -8.8 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 2 | 256 | 783.1 | 707.9 | -9.6 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 2 | 256 | 774.2 | 707.9 | -8.6 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 2 | 256 | 786.4 | 707.9 | -10.0 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 4 | 128 | 679.6 | 719.5 | +5.9 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 4 | 128 | 677.8 | 719.5 | +6.2 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 4 | 128 | 681.2 | 719.5 | +5.6 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 4 | 128 | 677.5 | 719.5 | +6.2 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 4 | 128 | 681.6 | 719.5 | +5.6 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 4 | 128 | 677.7 | 719.5 | +6.2 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 4 | 128 | 678.4 | 719.5 | +6.1 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 8 | 64 | 872.3 | 846.5 | -3.0 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 8 | 64 | 873.4 | 846.5 | -3.1 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 8 | 64 | 872.7 | 846.5 | -3.0 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 8 | 64 | 875.5 | 846.5 | -3.3 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 8 | 64 | 874.5 | 846.5 | -3.2 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 8 | 64 | 872.1 | 846.5 | -2.9 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 8 | 64 | 876.5 | 846.5 | -3.4 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 8 | 64 | 871.7 | 846.5 | -2.9 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 8 | 64 | 875.5 | 846.5 | -3.3 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 16 | 32 | 1313.1 | 1180.5 | -10.1 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 16 | 32 | 1312.4 | 1180.5 | -10.0 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 16 | 32 | 1312.0 | 1180.5 | -10.0 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 16 | 32 | 1311.2 | 1180.5 | -10.0 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 16 | 32 | 1308.7 | 1180.5 | -9.8 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 1 | 1024 | 1577.3 | 1450.6 | -8.0 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 1 | 1024 | 1574.7 | 1450.6 | -7.9 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 1 | 1024 | 1577.2 | 1450.6 | -8.0 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 2 | 512 | 1547.3 | 1459.9 | -5.6 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 2 | 512 | 1544.4 | 1459.9 | -5.5 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 4 | 256 | 1507.6 | 1452.8 | -3.6 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 4 | 256 | 1508.1 | 1452.8 | -3.7 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 4 | 256 | 1505.8 | 1452.8 | -3.5 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 8 | 128 | 1450.3 | 1602.5 | +10.5 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 8 | 128 | 1448.7 | 1602.5 | +10.6 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 8 | 128 | 1451.1 | 1602.5 | +10.4 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 16 | 64 | 2257.0 | 2252.5 | -0.2 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 16 | 64 | 2244.9 | 2252.5 | +0.3 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 32 | 32 | 3895.8 | 3705.0 | -4.9 |
| `matmul_row_tiling` | 4096 | 2048 | 8192 | 8 | 128 | 3041.3 | 3226.8 | +6.1 |
| `matmul_row_tiling` | 4096 | 4096 | 2048 | 1 | 512 | 1477.8 | 1398.5 | -5.4 |
| `matmul_row_tiling` | 4096 | 4096 | 2048 | 2 | 256 | 1371.9 | 1284.6 | -6.4 |
| `matmul_row_tiling` | 4096 | 4096 | 2048 | 4 | 128 | 1198.4 | 1296.1 | +8.2 |
| `matmul_row_tiling` | 4096 | 4096 | 2048 | 8 | 64 | 1541.9 | 1530.9 | -0.7 |
| `matmul_row_tiling` | 4096 | 4096 | 2048 | 8 | 64 | 1543.7 | 1530.9 | -0.8 |
| `matmul_row_tiling` | 4096 | 4096 | 2048 | 8 | 64 | 1542.5 | 1530.9 | -0.8 |
| `matmul_row_tiling` | 4096 | 4096 | 4096 | 1 | 1024 | 2664.6 | 2676.3 | +0.4 |
| `matmul_row_tiling` | 4096 | 4096 | 4096 | 2 | 512 | 2621.4 | 2575.5 | -1.8 |
| `matmul_row_tiling` | 4096 | 4096 | 4096 | 4 | 256 | 2588.3 | 2605.5 | +0.7 |
| `matmul_row_tiling` | 4096 | 4096 | 4096 | 8 | 128 | 2530.2 | 2919.8 | +15.4 |
| `matmul_row_tiling` | 4096 | 8192 | 2048 | 8 | 64 | 2966.6 | 2928.4 | -1.3 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 1 | 1024 | 1624.5 | 1582.0 | -2.6 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 1 | 1024 | 1618.7 | 1582.0 | -2.3 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 1 | 1024 | 1624.1 | 1582.0 | -2.6 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 1 | 1024 | 1626.1 | 1582.0 | -2.7 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 1 | 1024 | 1627.6 | 1582.0 | -2.8 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 2 | 512 | 1688.7 | 1424.2 | -15.7 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 2 | 512 | 1691.7 | 1424.2 | -15.8 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 2 | 512 | 1686.7 | 1424.2 | -15.6 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 2 | 512 | 1673.8 | 1424.2 | -14.9 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 4 | 256 | 1564.5 | 1403.4 | -10.3 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 4 | 256 | 1572.8 | 1403.4 | -10.8 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 4 | 256 | 1576.1 | 1403.4 | -11.0 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 4 | 256 | 1567.6 | 1403.4 | -10.5 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 8 | 128 | 1359.8 | 1434.4 | +5.5 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 8 | 128 | 1359.9 | 1434.4 | +5.5 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 8 | 128 | 1359.6 | 1434.4 | +5.5 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 8 | 128 | 1355.1 | 1434.4 | +5.8 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 16 | 64 | 1741.7 | 1684.0 | -3.3 |
| `matmul_row_tiling` | 16384 | 2048 | 2048 | 1 | 2048 | 3265.3 | 3108.1 | -4.8 |
| `matmul_row_tiling` | 16384 | 2048 | 2048 | 2 | 1024 | 3265.1 | 2934.6 | -10.1 |
| `matmul_row_tiling` | 16384 | 2048 | 2048 | 4 | 512 | 3371.4 | 2871.5 | -14.8 |
| `matmul_row_tiling` | 16384 | 2048 | 2048 | 4 | 512 | 3359.7 | 2871.5 | -14.5 |
| `matmul_row_tiling` | 16384 | 2048 | 2048 | 8 | 256 | 3098.2 | 2693.5 | -13.1 |
| `matmul_row_tiling` | 16384 | 2048 | 2048 | 8 | 256 | 3087.1 | 2693.5 | -12.7 |
| `matmul_row_tiling` | 16384 | 2048 | 2048 | 16 | 128 | 2712.9 | 2864.3 | +5.6 |
| `mm_nested_m_k` | 4096 | 2048 | 2048 | 1 | 512 | 846.3 | 781.2 | -7.7 |
| `softmax_noexp_row_tiling` | 8192 | 2048 | - | 8 | - | 641.0 | 673.7 | +5.1 |
| `softmax_noexp_row_tiling` | 16384 | 2048 | - | 16 | - | 1284.0 | 1347.4 | +4.9 |
| `softmax_noexp_row_tiling` | 16384 | 4096 | - | 16 | - | 2544.5 | 2694.7 | +5.9 |
| `softmax_row_tiling` | 2048 | 2048 | - | 4 | - | 171.7 | 168.4 | -1.9 |
| `softmax_row_tiling` | 2048 | 2048 | - | 8 | - | 204.4 | 222.6 | +8.9 |
| `softmax_row_tiling` | 2048 | 2048 | - | 16 | - | 323.5 | 356.6 | +10.2 |
| `softmax_row_tiling` | 2048 | 2048 | - | 32 | - | 525.6 | 571.3 | +8.7 |
| `softmax_row_tiling` | 4096 | 2048 | - | 1 | - | 389.8 | 320.0 | -17.9 |
| `softmax_row_tiling` | 4096 | 2048 | - | 1 | - | 263.0 | 320.0 | +21.7 ** |
| `softmax_row_tiling` | 4096 | 2048 | - | 4 | - | 352.6 | 336.8 | -4.5 |
| `softmax_row_tiling` | 4096 | 2048 | - | 4 | - | 351.5 | 336.8 | -4.2 |
| `softmax_row_tiling` | 4096 | 2048 | - | 4 | - | 351.1 | 336.8 | -4.1 |
| `softmax_row_tiling` | 4096 | 2048 | - | 8 | - | 360.6 | 336.8 | -6.6 |
| `softmax_row_tiling` | 4096 | 2048 | - | 8 | - | 359.0 | 336.8 | -6.2 |
| `softmax_row_tiling` | 4096 | 2048 | - | 8 | - | 360.3 | 336.8 | -6.5 |
| `softmax_row_tiling` | 4096 | 2048 | - | 8 | - | 360.1 | 336.8 | -6.5 |
| `softmax_row_tiling` | 4096 | 2048 | - | 16 | - | 401.2 | 445.2 | +11.0 |
| `softmax_row_tiling` | 4096 | 2048 | - | 16 | - | 400.4 | 445.2 | +11.2 |
| `softmax_row_tiling` | 4096 | 2048 | - | 16 | - | 405.4 | 445.2 | +9.8 |
| `softmax_row_tiling` | 4096 | 2048 | - | 32 | - | 645.7 | 713.2 | +10.5 |
| `softmax_row_tiling` | 4096 | 4096 | - | 2 | - | 710.8 | 747.5 | +5.2 |
| `softmax_row_tiling` | 4096 | 4096 | - | 4 | - | 674.6 | 673.7 | -0.1 |
| `softmax_row_tiling` | 4096 | 4096 | - | 8 | - | 690.9 | 673.7 | -2.5 |
| `softmax_row_tiling` | 6144 | 4096 | - | 2 | - | 1160.4 | 1191.6 | +2.7 |
| `softmax_row_tiling` | 8192 | 2048 | - | 2 | - | 761.6 | 747.5 | -1.8 |
| `softmax_row_tiling` | 8192 | 2048 | - | 4 | - | 731.8 | 673.7 | -7.9 |
| `softmax_row_tiling` | 8192 | 2048 | - | 4 | - | 727.4 | 673.7 | -7.4 |
| `softmax_row_tiling` | 8192 | 2048 | - | 8 | - | 665.5 | 673.7 | +1.2 |
| `softmax_row_tiling` | 8192 | 2048 | - | 8 | - | 667.8 | 673.7 | +0.9 |
| `softmax_row_tiling` | 8192 | 2048 | - | 8 | - | 667.4 | 673.7 | +0.9 |
| `softmax_row_tiling` | 8192 | 2048 | - | 8 | - | 665.6 | 673.7 | +1.2 |
| `softmax_row_tiling` | 8192 | 2048 | - | 8 | - | 666.4 | 673.7 | +1.1 |
| `softmax_row_tiling` | 8192 | 2048 | - | 8 | - | 665.8 | 673.7 | +1.2 |
| `softmax_row_tiling` | 8192 | 2048 | - | 8 | - | 669.0 | 673.7 | +0.7 |
| `softmax_row_tiling` | 8192 | 2048 | - | 8 | - | 666.4 | 673.7 | +1.1 |
| `softmax_row_tiling` | 8192 | 2048 | - | 8 | - | 668.7 | 673.7 | +0.7 |
| `softmax_row_tiling` | 8192 | 2048 | - | 16 | - | 681.1 | 673.7 | -1.1 |
| `softmax_row_tiling` | 8192 | 2048 | - | 16 | - | 676.0 | 673.7 | -0.3 |
| `softmax_row_tiling` | 8192 | 2048 | - | 32 | - | 856.1 | 890.3 | +4.0 |
| `softmax_row_tiling` | 8192 | 4096 | - | 2 | - | 1573.5 | 1658.8 | +5.4 |
| `softmax_row_tiling` | 10240 | 4096 | - | 2 | - | 2020.3 | 2144.1 | +6.1 |
| `softmax_row_tiling` | 12288 | 4096 | - | 2 | - | 2486.5 | 2644.2 | +6.3 |
| `softmax_row_tiling` | 16384 | 2048 | - | 1 | - | 4956.0 | 4287.4 | -13.5 |
| `softmax_row_tiling` | 16384 | 2048 | - | 2 | - | 1652.9 | 1658.8 | +0.4 |
| `softmax_row_tiling` | 16384 | 2048 | - | 4 | - | 1549.2 | 1495.0 | -3.5 |
| `softmax_row_tiling` | 16384 | 2048 | - | 4 | - | 1544.4 | 1495.0 | -3.2 |
| `softmax_row_tiling` | 16384 | 2048 | - | 4 | - | 1529.0 | 1495.0 | -2.2 |
| `softmax_row_tiling` | 16384 | 2048 | - | 8 | - | 1442.2 | 1347.4 | -6.6 |
| `softmax_row_tiling` | 16384 | 2048 | - | 8 | - | 1441.4 | 1347.4 | -6.5 |
| `softmax_row_tiling` | 16384 | 2048 | - | 8 | - | 1449.6 | 1347.4 | -7.1 |
| `softmax_row_tiling` | 16384 | 2048 | - | 16 | - | 1343.4 | 1347.4 | +0.3 |
| `softmax_row_tiling` | 16384 | 2048 | - | 16 | - | 1338.9 | 1347.4 | +0.6 |
| `softmax_row_tiling` | 16384 | 2048 | - | 16 | - | 1337.5 | 1347.4 | +0.7 |
| `softmax_row_tiling` | 16384 | 2048 | - | 16 | - | 1338.6 | 1347.4 | +0.7 |
| `softmax_row_tiling` | 16384 | 2048 | - | 32 | - | 1385.0 | 1347.4 | -2.7 |
| `softmax_row_tiling` | 16384 | 2048 | - | 32 | - | 1383.5 | 1347.4 | -2.6 |
| `softmax_row_tiling` | 16384 | 4096 | - | 1 | - | 9926.6 | 8574.7 | -13.6 |
| `softmax_row_tiling` | 16384 | 4096 | - | 2 | - | 9742.6 | 8005.5 | -17.8 |
| `softmax_row_tiling` | 16384 | 4096 | - | 2 | - | 9726.7 | 8005.5 | -17.7 |
| `softmax_row_tiling` | 16384 | 4096 | - | 4 | - | 3132.8 | 3317.6 | +5.9 |
| `softmax_row_tiling` | 16384 | 4096 | - | 4 | - | 3152.6 | 3317.6 | +5.2 |
| `softmax_row_tiling` | 16384 | 4096 | - | 8 | - | 2851.4 | 2990.0 | +4.9 |
| `softmax_row_tiling` | 16384 | 4096 | - | 8 | - | 2881.8 | 2990.0 | +3.8 |
| `softmax_row_tiling` | 16384 | 4096 | - | 16 | - | 2653.7 | 2694.7 | +1.5 |
| `softmax_row_tiling` | 16384 | 4096 | - | 16 | - | 2654.6 | 2694.7 | +1.5 |
| `softmax_row_tiling` | 16384 | 4096 | - | 16 | - | 2638.3 | 2694.7 | +2.1 |
| `softmax_row_tiling` | 16384 | 4096 | - | 32 | - | 2683.2 | 2694.7 | +0.4 |
