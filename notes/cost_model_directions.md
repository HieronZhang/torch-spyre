# Does the cost model point the right way?

The accuracy report (`cost_model_report.md`) asks how close the model's predictions are.
This asks something different and more practical: **would the model pick the right plan?**
A compiler using it never needs the absolute time — it needs to know that one coarse tile
size beats another, without running either.

Two studies. The first sweeps coarse tile size across every operation we have data for.
The second takes flash attention, the one realistic workload in the dataset, and asks why
it is mispredicted by a factor of 14 to 45.

The ladder tables come from `notes/verify_direction.py`. The diagnostic tables below them
are one-off decompositions, re-derived for this document and not yet scripted — treat their
individual cells as of their writing date, not as regenerated output. Everything is scored
with the live model; no hardware was used.

---

## 1. Choosing a coarse tile size

A **ladder** is one shape at one core count, measured at several tile counts. There are 26
of them in scope. For each, we ask what the model would have chosen and what that choice
would have cost.

Three numbers, in order of how much they matter:

- **Regret** — how much slower the model's choice is than the best available. This is what
  a compiler actually pays. Zero regret means the choice was tied for best, whether or not
  it was the same tile count.
- **Correct sign** — of the adjacent steps where the model predicts a change at all, how
  many move the way the measurement moves.
- **Flat and tied** — steps where the model predicts *exactly* no change, and ladders where
  several tile counts share the model's minimum. These are not wrong; they are blind.

| operation | ladders | pairs | correct sign | flat | picks best | median regret | worst regret | tied |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `matmul_row_tiling` | 17 | 56 | 44/56 | 0 | 11/17 | 0.0 % | 15.9 % | 0 |
| `softmax_row_tiling` | 8 | 29 | 17/17 | 12 | 5/8 | 0.0 % | 13.7 % | 6 |
| `bmm_3d2d_k_tiling` | 1 | 3 | 3/3 | 0 | 1/1 | 0.0 % | 0.0 % | 0 |
| **all** | **26** | **88** | **64/76** | **12** | **17/26** | **0.0 %** | **15.9 %** | **6** |

**The headline.** The model picks a tile count tied for best on **17 of 26** ladders. On
the other nine it costs **11.4 % on average, up to 15.9 %**. The median regret is 0.0 %, but
that number is arithmetically forced — with 65 % zeros the median can only be zero — so the
conditional figure above is the honest one. The distribution is bimodal: seventeen exact
zeros, then nothing until 2.2 %.

**Two caveats on the population.** Sixty of the 114 rungs rest on a single run, and 11 of
the 26 ladders are single-run throughout — including the two worst-regret ladders. Keeping
only rungs with repeats, and only ladders that still have three, leaves 13 ladders scoring
**7/13** with mean regret 4.7 %. The conclusion survives, but with far less margin than
17/26 suggests.

**The two operations fail differently**, and the difference matters because it points at
different fixes.

### Softmax is blind, not wrong

Every step where the softmax model predicts a change moves the right way — **17 of 17**.
But 12 of its 29 steps predict *exactly* zero change while the measurement moves by up to
9 %, and on 6 of its 8 ladders several tile counts share the model's minimum.

There are **two** causes, and neither is the one an earlier draft of this section named.

**Cause 1 — the spill derate saturates.** Take the 16384×2048 ladder at 32 cores:

```
 tiles  rows/core   underfill   spill   traffic MB   predicted   measured
     1        512      0.950    1.000        469.8        4287       4957
     2        256      0.950    0.812        134.2        1659       1654
     4        128      0.950    0.901        134.2        1495       1540
     8         64      0.950    1.000        134.2        1347       1446
    16         32      0.950    1.000        134.2        1347       1339   <- true best
    32         16      0.950    1.000        134.2        1347       1384
```

Read the columns carefully. The **underfill derate never moves** — it sits at its 0.95 cap
on every rung. The column that moves is the **spill derate**, and it is the sole reason the
prediction changes at all between 2 and 8 tiles. Once it reaches 1.000 at 8 tiles, nothing
left in the model depends on the tile count, and the three remaining plans become
indistinguishable. (An earlier draft of this section named the underfill derate as the
tile-sensitive term. That was exactly backwards, and its own table showed so.)

**Cause 2 — the element-throughput floor binds.** On five of the twelve flat pairs the
prediction is not merely insensitive, it is *clamped*: the fused-reduction floor of §20 in
the accuracy report is the binding constraint. For 4096×2048 at 16 cores — the worst
softmax regret at 13.7 % — the prediction is **347.2 µs at every one of its four rungs**,
exactly the floor value, so the ladder is flat by construction. Three of the four rungs at
8 cores are the same.

**What is *not* true:** traffic does not become tile-count-invariant the moment a softmax
is tiled. On 16384×4096 it falls 939.5 → 604.0 → 268.4 MB across t1 → t2 → t4, and that
t2→t4 step is the single largest correct call in the whole softmax set (predicted −58.6 %,
measured −67.7 %). Traffic goes flat only from t4 on.

Nor is the underfill derate always inert: it drops below its cap in three of the eight
ladders, as far as **0.280**. It simply does not move on the ladder quoted above.

*Deciding experiment:* a ladder that keeps rows per core below about 12 throughout, which
is where the underfill term is live. The one ladder in the current set that spans it
(2048×2048 at 32 cores, rows/core 16 → 2) is also the only softmax ladder with no flat
pairs at all — though on absolute accuracy it is only fifth of eight, so it is better at
*ranking*, not at predicting.

### Matmul turns up one rung too early

Matmul has no flat steps and no ties — it always has an opinion — but **12 of its 56 steps
move the wrong way**. Nine of them predict a rise where the measurement falls; the other
three do the reverse. The nine are the ones that matter, because they are what pushes the
predicted optimum to too few tiles:

```
8192×2048       c32  t4 → t8 :  measured −13.4 %   predicted  +2.2 %
8192×2048×2048  c32  t4 → t8 :  measured −13.7 %   predicted  +2.2 %
16384×2048×2048 c32  t8 → t16:  measured −12.1 %   predicted  +6.3 %
4096×2048       c32  t2 → t4 :  measured −13.5 %   predicted  +1.6 %
```

So the model's curve bottoms out earlier than the real one, and it picks too few tiles.
All four worst regrets are exactly this. Two qualifications: the wrong predictions are not
always small — one is +20.8 % — and three of the nine have measured falls of 1.0–1.8 %, at
or below the run-to-run noise floor, so their sign carries no information. Six of the
fifty-six steps are wrong by a margin that clearly exceeds noise.

**Which term does it?** The re-read charge — the cost of a loop-invariant operand being
fetched again on every pass. It is the only term here that grows *monotonically* with tile
count (the underfill and spill derates move too, and the matmul area-spill term shrinks,
which is what masks the re-read early on):

```
matmul_row_tiling 8192×2048, 32 cores
  t   measured   predicted   traffic MB   of which re-read MB
  1       1624        1582         75.5                   0.0
  2       1688        1424         83.9                   8.4
  4       1570        1403        100.7                  25.2
  8       1360        1434        134.2                  58.7   <- model turns up here
 16       1742        1684        201.3                 125.8
```

All of the traffic growth is re-read: the non-re-read base is constant at 75.5 MB across
the whole ladder. In *time* the charge is smaller than that byte share suggests — it is
levied at peak bandwidth, outside the derates, so at 8 tiles it is 18 % of the predicted
time rather than 44 % of the bytes. Decomposing the turn itself, the +31.0 µs step from 4
to 8 tiles is +75.6 µs of re-read against −44.6 µs from everything else.

(The 16-tile rung is shown for the mechanism but is single-run and out of scope; the
in-scope ladder is 1, 2, 4, 8.)

### What we did *not* do, and why

The obvious move is to turn the re-read charge down. Sweeping its coefficient looks like it
supports that:

| re-read scale | picks best | mean regret | `matmul_row_tiling` RMS |
|---:|---:|---:|---:|
| 0.00 | 20/26 | 1.8 % | 27.3 % |
| 0.50 | 16/26 | 3.6 % | 14.5 % |
| **0.85 (shipped)** | **17/26** | **3.9 %** | **7.7 %** |
| 1.00 | 17/26 | 3.9 % | 8.5 % |

Turning it off entirely appears to *improve* ranking — 20/26 instead of 17/26 — while
wrecking absolute accuracy. That reading is wrong, and the reason is worth recording.

**At scale 0 the model goes blind, and its apparent wins are the tie-break.** With the
charge off, 18 of the 26 ladders have several tile counts sharing the model's minimum, and
**12 of the 20 "correct" picks are decided by the tie-break rather than by the model** —
`min()` returns the smallest tile count, which often happens to be near-best. At the
shipped 0.85 only 6 ladders tie and only 3 picks are decided that way.

So the re-read charge is what gives the model any tile-count discrimination at all.
Removing it would trade a real signal for a scoring artifact. The honest conclusion is that
the term is **right in kind and somewhat too strong in degree**, and that no single
coefficient fixes both ranking and accuracy — which is consistent with the accuracy
report's finding that this term is entangled with the underfill derate.

*Deciding experiment:* a repeat-backed ladder at the four shapes where the sign flips. The
falls are 12–14 %, well above the run-to-run noise floor, but each rests on few runs;
confirming them is what would justify touching the coefficient.

---

## 2. Flash attention

Flash attention is the only realistic workload in the dataset: 23 modellable operations,
two levels of coarse tiling, rank-4 matmuls. It is mispredicted by **+1370 % to +4431 %**
across the seven scoreable records — the model says a kernel measured at 2.2 ms will take
64 ms.

That is not a rounding problem, it is the whole aggregate: those seven rows take the
overall error across all 1721 scored runs from **20.4 % to 225.5 %**. Excluding flash while
it is unmodelled is not housekeeping, it is a prerequisite for any aggregate meaning
anything.

### The extractor does parse it

This was the first thing to check, and the answer is yes. In the dumped IR
(`haoyang_logs/flash_attn_ir.log`, one configuration at Lq = 4096), the post-pass section
holds **23 `ComputedBuffer` operations, 17 of which carry coarse-tiling metadata** —
`loop_count = [8, 4]`, so 32 iterations, tiling two output dimensions and no reduction
dimension. The other six sit outside the loop. Features are extracted for all 23; nothing
bails out.

(One correction to earlier notes: the `flash_*.txt` paths referenced in the sweep labels no
longer exist. `haoyang_logs/ir/` still holds 54 `flashRS_*.txt` files, but they are
one-line summary stubs with no IR in them, so `flash_attn_ir.log` is the only usable dump.)

### One term causes essentially all of it

| shape | measured | predicted | error | rows/core | derate | prediction ÷ derate |
|---|---:|---:|---:|---:|---:|---:|
| Lq 1024, 4×2 tiles | 2245 | 64171 | +2759 % | 0.0312 | 0.0166 | 1063 |
| Lq 1024, 8×2 tiles | 2684 | 98941 | +3587 % | 0.0156 | 0.0103 | 1023 |
| Lq 1024, 8×4 tiles | 3742 | 158517 | +4137 % | 0.0078 | 0.0065 | 1023 |
| Lq 2048, 8×4 tiles | 7811 | 321672 | +4018 % | 0.0078 | 0.0065 | 2075 |
| Lq 4096, 8×4 tiles | 18922 | 661902 | +3398 % | 0.0078 | 0.0065 | 4270 |
| Lq 2048, k-tiled (32 ops) | 12601 | 570903 | +4431 % | 0.0039 | 0.0040 | 2299 |
| Lq 2048, 8×1 tiles | 11131 | 163634 | +1370 % | 0.0312 | 0.0166 | 2710 |

`rows/core` should be a row count in the hundreds. It is **0.008**. That feeds the underfill
derate, which divides the memory term by 60–248× across the seven records.

**Root cause**, in `dump_cost_model.py:521-526`:

```python
rows = out_dims[-2]                 # second-to-last DEVICE dimension
if out_mem != "lx":
    rows = rows / loop_trip
tile_rows_per_core = rows / _row_split(op, cores)
```

Both inputs are wrong for a rank-4 tensor:

1. `out_dims[-2]` assumes a rank-3 device layout, where the second-to-last axis really is
   the row count. Flash's rank-4 logical tensors have a **rank-5** device layout — the
   Lq = 4096 run's tiled matmul output is `logical=[1, 4, 1024, 128]` laid out as
   `dims=[4, 1024, 2, 1, 64]` — so `[-2]` picks a degenerate axis of size **1**.
2. `_row_split` picks the output variable with the largest write-index coefficient. For
   rank ≥ 3 that is the batch or head dimension, not rows. The matmul decoder already fixed
   exactly this mistake for itself (`dump_cost_model.py:315-320`); `_row_split` was never
   updated.

Worked example, from the recorded features of the Lq = 1024, 4×2-tile run: its tiled matmul
output is `logical=[1, 8, 512, 128]`, `dims=[8, 512, 2, 1, 64]`. So `dims[-2] = 1`, and with
`loop_trip = 8` and a row split of 4 that gives `1/8/4 = 0.03125` — exactly the recorded
`tile_rows_per_core`. The row extent it should have used, `logical[-2]`, is **512**.

### Fixing it is necessary but not sufficient

The last column shows what the model would say with the derate neutralised: 1063, 1023,
1023, 2075, 4270 against measured 2245, 2684, 3742, 7811, 18922. That is no longer 35× over
— it is **2 to 4.4× under**. (Simply dividing by the derate matches a proper forced
`eff = 1` re-score to three significant figures here, but only because four conditions
happen to hold on flash: the fill term is negligible, the element floor is structurally
disabled by the presence of a matmul, the re-read term is zero, and arithmetic is under
0.2 % of memory so the overlap contributes nothing. Each of the four secondary fixes below
would break one of them.) Note also that the first three rows share Lq = 1024 and differ
only in tile count, so the 2.1 → 2.6 → 3.7 progression there is a tiling effect; only
3.7 → 3.8 → 4.4 is size.

So the derate bug is the dominant error and must be fixed, but a second, opposite error is
hiding behind it. Four things contribute, all verified and none yet fixed:

- **Arithmetic is under-counted 8–64×.** `matmul_macs` is recorded per tile, and because
  flash tiles output dimensions rather than the reduction dimension, nothing rescales it by
  the loop trip count.
- **Bytes are under-counted.** The keys, values and mask are loop-invariant across the head
  and query nests, yet every argument records `loop_factor = 1`, so each is charged once for
  the whole 32-iteration nest. (The bundle de-duplication is *not* a second cause here —
  each external input is read by exactly one operation in this bundle, so the de-dup branch
  changes nothing. `loop_factor = 1` alone does it.)
- **The batched-matmul rate never applies.** Both layout detectors require exactly rank-3
  operands, so flash's rank-4 matmuls take the plain 2-D rate of 1140 rather than the
  both-default 161.6 — a further 7× arithmetic under-count on top of the one above. This is
  a deliberate gate, not an oversight: the code says so, and extending those rates to rank 4
  would extrapolate into a regime with no measurements.
- **The fused-reduction floor is unreachable.** It is gated to bundles containing no
  matmul, so the one term built for fused softmax chains never applies to flash — whose 23
  operations are mostly exactly that.

### The fix, applied

`tile_rows_per_core` now takes the row extent from the **logical** shape, which is the row
count at every rank, instead of `device[-2]`, which is the row count only at rank 2. One
line, in `dump_cost_model.py`.

It is safe by construction: across every recorded tiled operation, `device[-2]` and
`logical[-2]` are **equal on all 1177 rank-2 cases** — the entire population the model was
calibrated on — and differ only on the 74 rank-3 and rank-4 cases, which is exactly the set
that was broken. Gold categories are unchanged.

Simulating it on the seven stored flash records:

| shape | measured | before | after | error before | error after |
|---|---:|---:|---:|---:|---:|
| Lq 1024, 4×2 | 2245 | 64171 | 2368 | +2759 % | **+6 %** |
| Lq 1024, 8×2 | 2684 | 98941 | 3652 | +3587 % | +36 % |
| Lq 1024, 8×4 | 3742 | 158517 | 5850 | +4137 % | +56 % |
| Lq 2048, 8×4 | 7811 | 321672 | 13172 | +4018 % | +69 % |
| Lq 4096, 8×4 | 18922 | 661902 | 33370 | +3398 % | +76 % |
| Lq 2048, k-tiled | 12601 | 570903 | 96143 | +4431 % | +663 % |
| Lq 2048, 8×1 | 11131 | 163634 | 33926 | +1370 % | +205 % |

**RMS across the seven: 3521 % → 266 %.** Five of seven land within +6 % to +76 %; the
k-tiled variant and the single-query-tile run stay high, and they are the two whose
remaining error the four secondary effects above should account for.

**This cannot be banked from stored data.** The change is in the *extractor*, so it only
affects features produced from now on; every flash record in the database still carries the
old, broken value. The table above is a simulation, not a re-score.
`docs/source/user_guide/examples/run_flash_reextract.sh` re-runs the same seven
configurations to make it real, and states what to check before folding the log — including
whether the spill term now fires on flash for the first time, applying a matmul-calibrated
threshold to what is mostly a softmax chain.

Everything else here stays **recorded, not fixed**. Flash is a single workload, and fitting
terms to it is exactly the curve-fitting this project has refused throughout.

**Exclude it from the aggregates.** Flash is not filtered by `eval_model.in_scope()` — only
by prose in the accuracy report — so it lands in the `other` category and, as noted above,
takes the overall error from 20.4 % to 225.5 %. Seven rows out of 1721. An explicit
exclusion (or an `--exclude` flag) should go in before anyone reads an aggregate again.

---

## Reproducing

```bash
python3 notes/verify_direction.py            # summary and per-ladder table
python3 notes/verify_direction.py --pairs    # every wrong-way and flat step
python3 notes/verify_direction.py --markdown # the tables above
```
