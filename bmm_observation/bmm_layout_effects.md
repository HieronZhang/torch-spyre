# Operand memory layout sets the speed of a batched matmul on Spyre

A batched matmul on Spyre runs up to **8.3× faster or slower depending only on how its
three operands are laid out in memory**. Every configuration moves exactly the same
bytes and performs exactly the same arithmetic; the difference is the order of the
tensors' outer axes.

This note reports the full measurement — all eight combinations of the three operands'
layouts, at three batch sizes — and what it implies for the matmul preferred-layout
work.

## 1. Terms

**The three operands.** The measured operation is `torch.bmm(A, B) -> C`:

| operand | role | logical shape |
|---|---|---|
| `A` | first input | `[B, M, K]` |
| `B` | second input | `[B, K, N]` |
| `C` | output | `[B, M, N]` |

**The two layouts.** On Spyre a tensor's logical axes are mapped to a device layout
whose innermost axis is a 64-element stick. For a rank-3 tensor, which of the remaining
two axes is slowest-varying is a free choice, expressed as a `dim_order`. Both choices
are measured here:

| name | `dim_order` | device shape of `[4, 1024, 2048]` | slowest-varying axis |
|---|---|---|---|
| **row-outer** | `[0, 1, 2]` | `[1024, 32, 4, 64]` | row (`M`); the batch axis sits just inside the stick |
| **batch-outer** | `[1, 0, 2]` | `[4, 32, 1024, 64]` | batch (`B`) |

Row-outer is what the compiler produces by default. Under it, the elements belonging to
one batch element are strided across the whole tensor; under batch-outer each batch
element is contiguous.

Both orders keep the stick axis in place, so switching between them changes no byte
count and inserts no re-layout copy. That is what makes the comparison below a
measurement of layout alone.

## 2. Method

`torch.bmm` at `M = 1024`, `K = 2048`, `N = 1024` in fp16, on 32 cores, at batch sizes
2, 4 and 8. The two inputs are placed on the device with an explicit `dim_order`. The
output cannot be placed directly — the compiler chooses its layout — so it is requested
through `SPYRE_MATMUL_PREFERRED_LAYOUT="output"` and the layout that actually came back
is read off the result tensor and recorded per cell.

Each cell is the median of 7 profiled repeats of device kernel time; host transfers and
buffer initialisation are excluded. All 24 cells were measured in one session, on one
compiler build. That matters: kernel performance moves as the compiler develops, so a
cell taken from a different build would be indistinguishable from a layout effect.

## 3. Results

Times in microseconds. The last column is each cell divided by the fastest cell at
`B = 4`.

| A | B | C | B=2 | B=4 | B=8 | ÷ fastest (B=4) |
|---|---|---|---:|---:|---:|---:|
| row | row | row | 961 | 3792 | 7459 | 8.28× |
| row | row | batch | 944 | 3857 | 7589 | 8.42× |
| row | batch | row | 603 | 2276 | 4531 | 4.97× |
| row | batch | batch | 590 | 2340 | 4672 | 5.11× |
| batch | row | row | 706 | 2246 | 4530 | 4.90× |
| batch | row | batch | 687 | 2249 | 4526 | 4.91× |
| batch | batch | row | 337 | 777 | 1516 | 1.70× |
| **batch** | **batch** | **batch** | **228** | **458** | **917** | **1.00×** |

![All eight layout combinations at batch 2, 4 and 8, each divided by the fastest cell of
its own batch size. The spread reaches 8.4x. The output layout changes almost nothing
until both inputs are batch-outer, where it is worth a further 1.7x.](layout_cube.png)

## 4. What the numbers show

**Layout spans a factor of eight.** Between the slowest and the fastest arrangement of
the same computation: **8.3× at `B = 4`, 8.1× at `B = 8`**, and 4.2× at `B = 2`. The
compiler's default arrangement — row-outer everywhere — is the slowest of the eight.

**Both inputs have to be batch-outer, and they do not act independently.** At `B = 4`,
switching `A` alone is worth 1.69× and switching `B` alone 1.67×; if the two effects
combined independently, switching both would give about 2.8×. It gives **4.88×**. The
same pattern holds at `B = 8` (1.65×, 1.65×, 4.92×). Half-converting the inputs
therefore captures far less than half the available speedup.

**The output layout is worth nothing until both inputs are fixed, and then it is worth
1.7×.** With either input still row-outer, making `C` batch-outer moves the time by at
most 3 % in either direction, and not consistently: slightly faster at `B = 2`, slightly
slower at `B = 4` and `B = 8`. That is no effect. Once both inputs are batch-outer it is
worth **1.70× at `B = 4`**, 1.65× at `B = 8` and 1.48× at
`B = 2`. This is the strongest ordering constraint in the data: the output only pays
after the inputs have stopped being the limiter.

## 5. Implications for `matmul_preferred_layout`

The three settings map onto three cells of the table above. At `B = 4`:

| setting | cell it produces | time | vs. off |
|---|---|---:|---:|
| off (default) | A row, B row, C row | 3792 µs | — |
| `"output"` | A row, B row, C batch | 3857 µs | 1.7 % slower |
| `"on"` | A batch, B batch, C batch | 458 µs | **8.3× faster** |

Two conclusions follow for a batched matmul of this shape:

1. **The entire benefit is in `"on"`.** `"output"` alone does not help, and here it is
   marginally harmful — consistent with the general finding that the output only pays
   once both inputs are batch-outer.
2. **A partial rollout is worth much less than it looks.** Because the two inputs
   combine super-multiplicatively, converting one operand — or converting the output
   first — captures a small fraction of the 8.3×.

## 6. The same cube at one core

Every cell above was measured on 32 cores, where the layout under test and the work
division the compiler chose for it are multiplied together. At **one** core there is no
division to choose, so re-measuring the cube there separates the two. All eight cells, both
batch sizes, beside the 32-core column from §3:

<!-- BEGIN:cube_1core -->
| A | B | C | B=2 (µs) | B=4 (µs) | B=4 at 32 cores (µs) |
|---|---|---|---:|---:|---:|
| row | row | row | 5,502 | 21,322 | 3,792 |
| row | row | batch | 14,659 | 30,942 | — |
| row | batch | row | 4,151 | 12,828 | 2,276 |
| row | batch | batch | 4,140 | 12,698 | — |
| batch | row | row | 4,159 | 12,294 | 2,246 |
| batch | row | batch | 5,670 | 11,602 | — |
| batch | batch | row | 4,135 | 8,261 | 777 |
| batch | batch | batch | 4,135 | 8,261 | 458 |
<!-- END:cube_1core -->

**Only one of the three findings survives.**

| finding (§4) | at 32 cores | at 1 core | survives? |
|---|---|---|---|
| layout spans a large factor | 8.3× | **2.6×** | yes, but a third the size |
| the two inputs combine super-multiplicatively | 1.73× of independent | **0.90×** | **no** — they combine independently |
| the output layout is worth 1.7× once both inputs are batch-outer | 1.70× | **1.00×** | **no** — worth nothing |

### The inputs

<!-- BEGIN:effects -->
| cores | switch A alone | switch B alone | if independent | measured together | vs independent |
|---|---:|---:|---:|---:|---:|
| 1 | 1.73× | 1.66× | 2.88× | 2.58× | **0.90×** |
| 32 | 1.69× | 1.67× | 2.81× | 4.88× | **1.73×** |
<!-- END:effects -->

Each operand is worth about the same at either core count — switching `A` alone buys 1.73×
at one core and 1.69× at 32; `B` alone 1.66× and 1.67×. **What changes is how they combine.**
At 32 cores the pair is worth 1.73× more than two independent effects would give; at one core
it is worth 0.90×, which is independence within measurement error. The
super-multiplicativity is not a property of the layouts. It appears only when the work is
divided.

### The output

<!-- BEGIN:output_layout -->
| A | B | C row-outer (µs) | C batch-outer (µs) | worth |
|---|---|---:|---:|---:|
| row | row | 21,322 | 30,942 | 0.69× |
| row | batch | 12,828 | 12,698 | 1.01× |
| batch | row | 12,294 | 11,602 | 1.06× |
| batch | batch | 8,261 | 8,261 | 1.00× |
<!-- END:output_layout -->

At one core, asking for a batch-outer output **never helps**. With both inputs already
batch-outer it changes nothing at all (8,261 against 8,261 µs). With the inputs left
row-outer it is actively harmful — 1.45× slower at `B = 4` and 2.66× at `B = 2`. The 1.70×
that §4 reports is therefore also a work-division effect, not a layout one.

### Where the missing factor lives

The two figures reconcile exactly. Pure layout is worth 2.58×, and the best layout also
*parallelises* 1.89× better than the worst — 10.6× speedup from 1 to 32 cores against 5.6×:

```text
2.58×  (layout, measured at one core)
1.89×  (the best layout scales better across 32 cores)
-----
4.88×  = the 32-core input effect measured in §3
```

Adding the output layout's 1.70×, which is likewise a many-core effect, gives §3's 8.3×. So
**roughly a third of the headline factor is layout, and two thirds is layout changing how
well the work divides.**

### The work-division ladder

Holding `B` row-outer and sweeping the core count:

<!-- BEGIN:ladder -->
| cores | A row-outer (µs) | A batch-outer (µs) | layout worth | speedup vs 1 core |
|---:|---:|---:|---:|---:|
| 1 | 21,475 | 12,252 | 1.75× | 1.0× |
| 2 | 18,572 | 14,253 | 1.30× | 1.2× |
| 4 | 11,710 | 7,910 | 1.48× | 1.8× |
| 8 | 6,591 | 3,650 | 1.81× | 3.3× |
| 16 | 3,784 | 2,229 | 1.70× | 5.7× |
| 32 | 3,804 | 2,252 | 1.69× | 5.6× |
<!-- END:ladder -->

The layout advantage is 1.69–1.81× at every core count except two, so switching `A` is worth
about the same whatever the division. But the scaling is poor in absolute terms: 32 cores buy
only 5.4–5.7× over one core, and **beyond 16 cores nothing improves at all** — 16 and 32 cores
measure the same to within 0.5 %. Whatever limits scaling here is not the layout.

## 7. Reproducing

`run_layout_cube.sh` in this directory re-measures the whole cube. It is self-contained:
it needs Spyre hardware, `torch` and `torch_spyre`, and the preferred-matmul-layout
support (upstream PR #3364), which is what provides the `C` axis.

```sh
./run_layout_cube.sh                     # B = 2, 4, 8 at 1024x2048x1024, as above
BATCHES="4" ./run_layout_cube.sh         # a single batch size
M=2048 K=2048 N=1024 ./run_layout_cube.sh
REPS=15 ./run_layout_cube.sh             # more repeats per cell
```

It writes one `SUMMARY` line per cell to `layout_cube.log`. Before reading a result off
it, check that `layout_c` really differs between the two rows that share a `layout_a`
and `layout_b`; if it does not, the output preference was not honoured on that build and
those two rows are the same experiment rather than a comparison.
