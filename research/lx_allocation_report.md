# Cost Model Verification using Flash-attn with Diffrernt Configs and Different LX Allocation Policies.

Spyre gives each core 1587 KB of LX scratchpad. When a fused kernel's intermediate buffers
do not all fit, the compiler (LX planning) picks which ones stay on chip and which spill to HBM. On the
measurements below **that policy choice is worth 17.4 % and 26.4 % of kernel time, and the
compiler's default policy picks the slower allocation on every shape where the choice
matters.**

The cost model is useful for choosing the policy (although the `cpsat` is already doing very good), within limits this note is specific about. It correctly predicts the relative performance among
**15 of 15** pairs of coarse-tiled flash-attention variants correctly. It also correctly predicts the relative performance of
**95 of 103** and **61 of 65** pairs of shape variants of flash-attn at 32 and 8 cores.

## 1. Terms

**LX** is a per-core software-managed scratchpad, 1587 KB usable. A buffer that fits is read
and written on chip; one that does not is *spilled*, and every access costs HBM bandwidth.

**An allocation** is the subset of a kernel's intermediates that stay in LX. Only buffers
produced *and* consumed inside the kernel are eligible: a graph input must be read from HBM
and an output written back, so neither is a choice.

**The policies** are `layout_solver` settings. Three appear below:

| policy | how it chooses |
|---|---|
| `greedy` (**default**) | no objective: walks the program in order, keeps each buffer that fits |
| `bestfit` | orders by `(lifetime - discount) / uses`, then packs in that order |
| `cpsat` | exact solver maximising retained `(reads + 1) x size` -- a **byte** objective |

<small>
Two more ship and are not measured here, for different reasons. <code>simulated_annealing</code>
is stochastic, so it is not a reproducible arm — two runs of one configuration can allocate
differently, which the method below depends on not happening. <code>firstfit</code> shares
<code>bestfit</code>'s ordering and differs only in which gap it places a buffer into; that was
assumed not to change <em>which</em> buffers stay resident, but it can — the two leave different
fragmentation behind, so a later buffer may find room under one and not the other.
</small>

**`optimum` is not a policy.** It is the allocation with the lowest *predicted* time (using the cost model) among
all that fit, found by exhaustive search. It is the reference the other rows are scored
against.

## 2. Method

Coarse tiling compiles on the 2026-07-30 build (torch 2.11) and does not on the current one
(torch 2.13). That is why the measurements come in two sets, and why the tiled ones are old:
they cannot be retaken.

**Tiled** (section 3, last table): six flash configurations varying the tile counts, on the
2026-07-30 build.

**Untiled** (everything else): the current build. Head count, sequence length and core count
were swept, keeping shapes where one buffer fits the budget but the working set does not --
the condition for the allocator to have a choice. Four of the nine that compiled qualified.

**How a policy is evaluated.** For each shape, every allocation that fits is enumerated
offline, and the one each policy would choose is identified from its published objective.
Those sets become the arms: the program is compiled once per distinct set, with the compiler
forced to use that set, and measured. The cost model scores the same sets. So a policy is
never run -- what is compared is the allocation it would produce, priced two ways, on one
program at one shape and core count.

Forcing the set needs one addition to the LX planning pass: a check at the point where the
allocator decides residency, where `LX_FORCE_ONLY` names the buffers allowed in LX and
everything else is refused. It is inert unless the variable is set, and each run's residency
is read back and checked against what was requested.

<small>Kernel time moves as the compiler develops, so no number from one build is set beside a
number from the other. Each figure is the mean of four rounds, interleaved in alternating
order; round-to-round spread is 0.4-1.8 %.</small>

## 3. Results

The same program, shape and core count within each block -- only the LX set differs. Each
shape has 22 movable buffers, so between 262,144 and 2,097,152 allocations satisfy the
capacity constraint; the reference row is the best of all of them, found by exhaustive
search. That is affordable because peak footprint is monotone in the allocation -- if a set
does not fit, no superset fits -- so one depth-first walk pruning on first infeasibility
visits only the feasible family.

<!-- BEGIN:alloc_213 -->
| flash shape | allocation | keeps | measured (µs) | predicted (µs) | order |
|---|---|---:|---:|---:|:--:|
| `H=4 Lq=2048 Lk=2048 sencores=32 (untiled)` | optimum+cpsat | 20 | 1,869.8 | 1,413.3 | ✓ |
|  | greedy+bestfit | 20 | 2,195.5 | 1,860.7 | ✓ |
| | _model orders these_ | | | | **correctly** |
| `H=16 Lq=1024 Lk=1024 sencores=32 (untiled)` | optimum+cpsat | 20 | 1,953.4 | 1,431.5 | ✓ |
|  | greedy+bestfit | 20 | 2,468.2 | 1,878.9 | ✓ |
| | _model orders these_ | | | | **correctly** |
| `H=8 Lq=512 Lk=512 sencores=8 (untiled)` | bestfit | 17 | 617.0 | 298.5 | →4 |
|  | cpsat | 17 | 645.1 | 259.1 | ✓ |
|  | optimum | 16 | 715.0 | 256.0 | →1 |
|  | greedy | 15 | 732.5 | 272.5 | →3 |
| | _model orders these_ | | | | **WRONG** |
<!-- END:alloc_213 -->

Ranking whole configurations, rather than allocations within one. Eight of these shapes
were measured in both sessions and agree to 0.0-1.0 %; each is counted once, at the mean of
its repeats.

<!-- BEGIN:ranking_213 -->
| core count | configurations | concordant pairs | Kendall τ | predicted ÷ measured |
|---|---:|---:|---:|---|
| 32 | 15 | 95/103 | **+0.84** | 0.61–1.03× (mean 0.85) |
| 8 | 12 | 61/65 | **+0.88** | 0.29–0.75× (mean 0.55) |
| both pooled | 27 | 314/348 | +0.80 | 0.29–1.03× |
<!-- END:ranking_213 -->

Coarse-tiled flash on the older build, sorted by measured time:

<!-- BEGIN:ranking_211 -->
| tile configuration | measured (µs) | predicted (µs) | predicted ÷ measured |
|---|---:|---:|---:|
| `H=32 Lq=1024 Lk=1024 htiles=4 qtiles=2` | 2,245 | 5,751 | 2.6× |
| `H=32 Lq=1024 Lk=1024 htiles=8 qtiles=2` | 2,684 | 5,932 | 2.2× |
| `H=32 Lq=1024 Lk=1024 htiles=8 qtiles=4` | 3,742 | 6,358 | 1.7× |
| `H=32 Lq=2048 Lk=2048 htiles=8 qtiles=4` | 7,811 | 12,901 | 1.7× |
| `H=32 Lq=2048 Lk=2048 ktiles=2` | 12,601 | 15,317 | 1.2× |
| `H=32 Lq=4096 Lk=4096 htiles=8 qtiles=4` | 18,922 | 26,547 | 1.4× |
| **15 of 15 pairs ordered correctly** (τ = +1.00) | | | **1.2–2.6×** |
<!-- END:ranking_211 -->

![Every movable buffer of one contested kernel, drawn across the operations it is live for,
with height proportional to per-core size. Grey buffers are kept by both policies; the four
coloured ones are the entire disagreement. They are all 1024 KB and differ only in how often
they are read -- and in each pair the once-read buffer is produced first, which is what a
program-order policy reaches first.](lx_allocation.png)

## 4. What the numbers show

**The default policy is 17 % and 26 % slower, and both allocations are the same size.** On
the two 32-core shapes the winning and losing sets each keep 20 buffers, all 1024 KB per
core. They differ in *which* four:

| | `H=16 Lq=Lk=1024`, 32 cores |
|---|---|
| movable buffers | 22 |
| kept by both policies | 18 |
| bytes resident, either way | 3648 KB |
| **buffers they disagree on** | **4** |

| buffer | per-core | times read | byte weight | `cpsat` | `greedy` |
|---|---:|---:|---:|---|---|
| `b15` | 1024 KB | **2** | 3072 K | **keep** | spill |
| `b8` | 1024 KB | **2** | 3072 K | **keep** | spill |
| `b14` | 1024 KB | 1 | 2048 K | spill | **keep** |
| `b7` | 1024 KB | 1 | 2048 K | spill | **keep** |

Spilling costs one write plus one read per consumer, so keeping a twice-read buffer saves
three transfers and a once-read buffer two. Same footprint, same residency, different reuse --
which size alone cannot see. Retained byte weight is **9536 K against 7488 K**, a ratio of
1.27x; the measured times differ by 1.26x. The default walks the program in order, meets `b7`
and `b14` first, takes them, and has no room for the pair directly behind them worth 1.5x
more.

**The byte objective is beatable, by 4.6 %.** On the third shape `cpsat`'s allocation
measured 645.1 us against 617.0 us for `bestfit`'s -- two measured times, not a model
estimate. Against the 17 % and 26 % the default costs, it is second-order.

**The model does not rank allocations.** Section 3's question -- given one program at one
shape, which LX set is fastest -- is answered correctly on 4 of 8 pairs:

<!-- BEGIN:alloc_pairs -->
| flash shape | allocations | pairs | correct | order |
|---|---:|---:|---:|:--:|
| `H=4 Lq=2048 Lk=2048 sencores=32 (untiled)` | 2 | 1 | 1 | ✓ |
| `H=16 Lq=1024 Lk=1024 sencores=32 (untiled)` | 2 | 1 | 1 | ✓ |
| `H=8 Lq=512 Lk=512 sencores=8 (untiled)` | 4 | 6 | 2 | **wrong** |
| **total** | | **8** | **4 of 8** | **2 of 3 shapes** |
<!-- END:alloc_pairs -->

Two of the three shapes are ordered correctly, but each contributes a single pair; the third
supplies six of the eight and fails four. Coin-flipping scores 4 of 8. **On this evidence the
model cannot be used to choose an allocation.** Section 5 shows what is and is not known
about why.

**It does rank whole configurations well**, which is a different question -- comparing
programs or shapes against each other rather than allocations within one. 95 of 103 pairs at
32 cores and 61 of 65 at 8 over 27 distinct shapes; all 15 pairs of coarse-tiled variants at
1.2-2.6x of measured. Across core counts it does not -- section 5.

**The predicted times are low but the predicted gap is high.** Predictions sit at 0.72-0.76x
of measured, yet the *difference* the model attributes to the allocation is +31.7 % and
+31.3 % against 17 % and 26 % measured. It over-states what a spill costs while
under-stating the kernel. Two points do not identify a cause; the suspect is that a spill is
charged at full HBM cost while the hardware overlaps part of it with compute, and the test is
to vary arithmetic intensity at a fixed allocation difference.

## 5. What does not hold

**The model's own optimum is not the fastest allocation.** On the third shape it ranks
`optimum` first and `bestfit` last; measured, `bestfit` is fastest and `optimum` third.

The cause is *not* a monotonicity failure, and an earlier draft of this note said it was.
Checked directly on this bundle: no single buffer can be added to any allocation without
predicted time falling or the capacity constraint biting -- zero violations. The two sets are
also not nested. `bestfit` keeps 17 buffers and `optimum` 16, but `bestfit` adds five that
`optimum` lacks and drops four that it has; both are maximal under the budget. They are
incomparable, and the model simply ranks them the wrong way round.

What it gets wrong is the traffic. The model scores `bestfit`'s set at 34.6 MB against
`optimum`'s 27.8 MB and concludes it is slower; the device runs it 14 % faster. So the error
is in what a given spill is priced at, not in any discontinuity. **The mechanism is not yet
identified.** It would be isolated by measuring the two sets' HBM traffic directly rather
than inferring it, which the profiler can report and this study did not collect.

**Predictions do not compare across core counts, because the hardware steps.**

<!-- BEGIN:cores_ladder -->
| cores | H=4 L=1024 measured | H=8 L=512 measured | H=16 L=512 measured | mean predicted ÷ measured |
|---:|---:|---:|---:|---:|
| 1 | 3,632 µs | 2,044 µs | 3,463 µs | **0.30×** |
| 2 | 3,613 µs | 2,018 µs | 3,762 µs | **0.30×** |
| 4 | 1,582 µs | 486 µs | 1,584 µs | **0.56×** |
| 8 | 1,565 µs | 480 µs | 1,592 µs | **0.56×** |
| 16 | 1,604 µs | 486 µs | 1,573 µs | **0.56×** |
| 32 | 276 µs | 194 µs | 333 µs | **0.85×** |
<!-- END:cores_ladder -->

One and two cores measure alike; four, eight and sixteen measure alike; only 32 changes
anything. The device is not using the extra cores across those plateaus, and the model
predicts a nearly flat value across 4-16 as well -- but a different one, so the ratio sits
level and wrong inside each plateau.

A separate investigation of the same gap on coarse-tiled softmax narrows it: at matched tile
geometry the achieved rate is 0.60x at 16 cores and 0.44x at 8, the per-core tile is
identical, and tile count is ruled out. It is **specific to fused reductions** -- a pointwise
kernel at 8 cores misses by only 13 %. It is deliberately not modelled: the deciding
experiment is a core ladder with repeats, and the present cells are single runs.

**Everything here is flash attention** -- three shapes, one program family. The case studies
meant to test other programs produced invalid predictions: several distinct programs reported
identical predicted times, the signature of a multi-graph compile where the feature dump
captures only the last graph. They need re-running.

## 6. A defect found and fixed while writing this

Coarse-tiled predictions were originally **29-45x above measured**. The model derated a tiled
kernel's memory term by a pipeline-fill efficiency keyed on the per-core tile HEIGHT alone,
`h = ROWS / (cores * tiles)`, fitted over `h = 2..32`. These records carried `h = 0.25` and
`0.0078` -- below one row per core, which the hardware cannot be in -- and the unbounded power
law kept going, reaching a 155x derate.

Two things were wrong, and only the first was obvious. The surface was being evaluated far
outside its support; and **height was the wrong variable.** Backing the derate out of 48 tiled
softmax rows at 32 cores shows the governing quantity is the per-core tile SIZE, not its
height -- the refitted surface is a single power product,

```text
eff = min(1.08, (h / 7.9)^0.50 * (COLS / 2048)^0.38)
```

anchored at the corner of its support and continued below. A narrow tensor never reaches the
cap at any height measured, which a height-only term cannot express. Tiled matmuls were split
onto the previous rows-only curve, frozen, because every row behind the new surface is a
softmax row.

The refit is reported in `cost_model_report.md` §16. Its effect here: **softmax RMS 28.4 ->
20.7 %** with no other category moving, and on these six flash points the absolute error falls
from 29-45x to **1.2-2.6x** with all 15 pairs still ordered correctly.

## 7. What follows

1. **Change the default `layout_solver` from `greedy` to `cpsat`** -- one configuration line,
   worth 17 % and 26 % here, never worse on any shape measured, independent of the cost model.
2. **Find why a spill is mispriced** before using the model to choose an allocation.
   Measure the two allocations' actual HBM traffic on the third shape and compare it with
   the 27.8 MB / 34.6 MB the model assumes.
3. **Do not fit the core-count gap yet** -- run the core ladder with repeats first. Until
   then, compare predictions only within one core count.

## 8. Reproducing

```sh
python3 research/probe_untiled_flash.py --analyse-only   # solve every allocation exactly
python3 research/emit_forced_allocations.py \
    --records research/untiled_flash_records.json --op flash_attn
python3 research/run_lx_experiments.py \
    --records research/untiled_flash_records.json --phases 1
python3 research/gen_lx_report.py                        # regenerate tables and figure
```

Allocations are pinned by name rather than selected by policy: several policies often choose
the same set, and equal runtimes would then say nothing about either. `LX_FORCE_ONLY="b8,b15,..."`
restricts LX eligibility to the named buffers, and the residency the compiler produced is read
back and checked against what was requested. Check the residency in the run's feature dump before reading a result: if
the requested buffers are not the ones that landed in LX, the rows are the same experiment
rather than a comparison.
