# Deriving a Cost Model for the IBM Spyre Accelerator

*An analytical model that predicts kernel runtime on Spyre from the loop-level IR, without
running it. This is the derivation: each term starts from an observation in the measurement
data, states the question it raises, and is settled by an isolation experiment before being
given a form. Every figure and accuracy number is regenerated from the live model against
`tools/cost_model/sweep_records.json`.*

---

## The ops we test, and the model at a glance

The model in full, with each term linked to the section that derives it. Parts I–IV give the
route: what was observed, what it raised, and what settled it.

### The data every number here is scored against

**2828 measurements recorded, 2074 in scope, 1610 scoreable.** Every accuracy figure in this report
is recomputed from the live model against `tools/cost_model/sweep_records.json` — nothing is hand-typed.
Regenerate the per-section accuracy lines and tables with `python3 tools/cost_model/report_tables.py`, and the
figures with `python3 tools/cost_model/plot_report.py`.

The three counts differ for reasons worth stating. *In scope* applies the permanent exclusions
below. *Scoreable* additionally requires that a record carry enough information to be re-costed by
the **current** model — either the logged per-op features, or an I/O block the features can be
reconstructed from and verified against. Runs made before feature logging existed can still be
compared against the prediction baked into them at the time, but they cannot test a model that has
since changed, so they are excluded from every accuracy figure here.

Measured time is each record's `kernel_us`. (Runs that repeat a configuration also store a
`kernel_us_min`; it exists on only ~45 % of rows, so using it would quietly reduce every
population to a repeat-backed subset.)

Three classes of measurement are **permanently excluded**, so that a number quoted anywhere in
this report always refers to the same population:

| excluded | why |
|---|---|
| fewer than 8 cores | never a target regime, and the few points were confounded — core count moved together with per-core area and split fanout, so they could only have been fitted, not explained |
| a fused reduction narrower than 1024 columns | the reduced-axis length dominates that rate; the short-row corner is a different regime and drags the shape term toward a case we do not care about |
| two early logs with corrupted features | the same measured configuration is recorded with a 16× different per-core row count than in every later log. The *measurements* agree to 0.8 %, so only the features drifted — such rows cannot judge the model in either direction |
| a coarse-tiled matmul whose work division is 16×2 or 2×16 | we do not choose that split — the coarse-tiling hint makes the planner pick it, and it sits far outside the point where the matmul/bmm compute rate was calibrated (one split, 4×8). A plain matrix multiply at the same splits is modelled well (10.8 % over 35 runs) and is deliberately kept |

**Accuracy of the single-operation model** (Parts I–III; coarse tiling is scored separately
in Part IV):

| section | n | RMS % | mean % | beyond ±10 % |
|---|---:|---:|---:|---:|
| Part I — single pointwise | 89 | 3.6 | −1.5 | 4 |
| §4 — broadcast / write | 281 | 5.7 | −0.2 | 22 |
| §6 — reduction | 97 | 7.2 | +3.0 | 14 |
| §7 — transport | 203 | 6.1 | −0.0 | 15 |
| §13 — matmul split shape | 318 | 15.1 | −0.3 | 89 |
| §14 — batched matmul | 135 | 33.0 | −16.7 | 49 |

The memory-bound ops (Parts I–II) are comfortably inside the ±15–20 % bar. Matmul is not yet:
§13's spread is two-sided (−48…+48 %), and §14 is the batched multiply at the only operand
layout the compiler emits — the faster order is measured but deliberately not modelled.

Two gaps worth stating plainly. **§8–11 has no scoreable data**: all 21 plain `mm` records
predate feature logging and carry no reconstructable I/O block, so the memory, compute and
overlap terms those sections derive are in practice validated through `mmwd` (§13) rather than
through `mm` itself. And there is no plain-`mm` IR dump on disk. Both are worth closing.

### The harness ops (what each benchmark actually runs)

| group | ops | torch expression |
|---|---|---|
| pointwise | `neg` `gelu` `exp` `mul` `add` `copy` | `-x`, `gelu(x)`, `exp(x)`, `a*b`, `a+b`, `x+1.0` (`copy` is a broadcast op) |
| reduction | `sumrow` `sumcol` `amax` `mean` `sumall` `read` | `sum(x,dim=1)`, `sum(x,dim=0)`, `amax(x)`, `mean(x,dim=1)`, `sum(x)`, `x` (pure read) |
| broadcast | `bcast` `mulbcast` `bcastcol` `write` | `a[R,C]+b[1,C]`, `a[R,C]*b[1,C]`, `a[R,C]+b[R,1]`, `b[1,C]+c[R,1]` |
| transport | `transpose` `transpose_outer` `cat0` `cat1` | `[R,C].transpose(0,1)` (the stick dimension is swapped into a row); `[R,8,C].transpose(0,1)`→`[8,R,C]` (swap the two **outer** axes, the innermost dimension `C` stays whole); `cat([x,x],dim=0/1)` |
| matmul | `mm` `mmwd` | `a@b`, with the work division either chosen by the compiler (`mm`) or forced by the harness (`mmwd`) |
| coarse-tiling | `softmax_row_tiling` `matmul_row_tiling` | `softmax(x,dim=-1)`, `a@b` — split into a loop so each step's working set fits in LX |

### One term to fix

**Effective bandwidth** here means bytes moved divided by time taken. It is not a hardware
constant: it is what a particular operation *achieved*, and most of this report is about why
it varies. Traffic is counted in the padded on-device layout rather than the logical tensor
size, because the padding is moved too.

### The model — full form

Per kernel, with `R` and `W` the bytes read from and written to HBM. Every term is shown; each is 0 / 1 when it
does not apply. The right column names the section where it is derived.

```text
T   = max( compute + mem − γ·min(compute, mem) + split ,  elem_floor )
                                                      mem = HBM / (eff · s_lx)

  HBM     = [ (R+W)/BW + α·min(R,W) ]  +  spill  +  write_extra  +  reread
  compute = MACs / cores / (peak · pt_eff)
  s_lx    = min(1, (cap / ws)^0.15)     for a coarse-tiled kernel with ws > cap    (else 1)
```

| term | form | derived in |
|---|---|---|
| `(R+W)/BW` | `BW` = 150 GB/s (pointwise and matmul); `BW_red(ROWS)=min(150, 114+61·e^(−ROWS/3700))·g(cores)` (row reductions; `g` is a derate for running on fewer than 32 cores, and is 1 on the full machine); per-op `BW_eff` for access-pattern ops | §1, §6, §7, §9 |
| `α·min(R,W)` | `α = 0.00574 ns/B` — read↔write bus **turnaround** (0 for one-directional traffic) | §2 |
| `spill` | `(A_bytes+B_bytes)·f(area)`, `area=(M/m)·(N/n)`, `f=min(1.5, max(0, 0.45·log₂(area/65536)))` — matmul operand **re-read** when the per-core output tile overflows on-chip capacity | §12 |
| `split` | `max(0,area−a₀)·[c_L·max(0,log₂(fan_long/8)) + c_S·max(0,log₂(fan_short/16))]`, `area=(M/m)·(N/n)`, `a₀ = 131072 elements`, `c_L = 2.6e−3`, `c_S = 2.9e−3 µs per element` — extra matmul operand re-read when a **large** per-core tile is **also** split many ways; two-sided (splitting the longer output dim bites sooner than the shorter); 0 for balanced or small tiles | §13 |
| `write_extra` | `min(2.0e-9·ROWS^1.75·COLS^2.6, 2.4·out_bytes)` (÷BW) — `write` outer-product, empirical + capped | §5 |
| `compute` | `multiply-accumulates / cores / (peak · pt_eff)`, `peak = 1140 per nanosecond per core`; a batched matmul (B≥4, cores≥8) in the compiler's default operand layout takes `peak = 160` instead; 0 for non-matmul | §10, §14 |
| `pt_eff` — **array fill** (derates arithmetic) | how full the arithmetic array is: `min(1,(rows/64)^0.35)` (`rows` = per-core rows); a coarse-tiled matmul's extra per-tile underfill is flagged, not modeled (`pt_eff=1`) | §10, §17 |
| `eff` — **pipeline fill** (derates memory) | `min(0.95, (h/13)^0.68)`, `h = per-core tile height = ROWS/(cores·tiles)` — how full the memory pipeline is, for a coarse-tiled memory-bound kernel | §17 |
| `s_lx` — **spill derate** (derates memory) | `min(1, (cap/ws)^0.15)` for `ws > cap`, `ws` (the working set one core holds) `= 2·(rows per core)·COLS·2 bytes`; `cap` = 512 KB for a tiled reduction, 2 MB for a tiled matmul — a per-core tile too large for LX moves at a lower rate | §18 |
| `γ·min(compute,mem)` | `γ = 0.46` — how much arithmetic and memory **overlap** (0 when there is no arithmetic) | §11 |
| `moves` (scales **every** arg) | `Π over loop levels of (the loop's trip count if the tensor's address does not vary along the dimension that level tiles, else 1)` — how many times a coarse-tiled loop transfers each tensor; 1 when there is no loop | §15 |
| `reread` | `size · (moves−1) · 0.85` — the repeated pass a loop-invariant operand makes, charged at peak and exempt from `eff`/`s_lx` | §16 |
| `elem_floor` | `elements / (cores · 1.51 per nanosecond)` — a fused reduction cannot finish faster than its element count allows; a **floor**, never binding at 32 cores | §19 |

Per-operation effective bandwidth, in GB/s: transpose `116` (flat); the block-shuffling transports `cat0`/`transpose_outer`/
`cat1` share `clamp(a − b·log₂(C/64) − d·log₂R, floor, peak)` (falls with the number of sticks per row, `C/64`; §7); summing down each column `113`; the broadcast operations `118`. The compiler never divides the summed dimension across cores.

---

## Part I — Pointwise: the gold baseline

### §1. Pointwise kernels are memory-I/O bound

Pointwise ops are the simplest kernels on the device: read one or more tensors, apply an
elementwise function, write one tensor. We use them to establish the reference every later
op is measured against.

**Traffic** throughout means the bytes a kernel actually transfers to or from HBM.

**Observation 1 — time is linear in bytes, with no fixed cost.** Sweeping `neg`, `gelu`
and `exp` (each reads one tensor and writes one) across sizes, kernel time is a straight
line in traffic that passes through the origin:

| op | R×C | HBM traffic | kernel time |
|---|---|---|---|
| `neg` | 2048×1024 | 8.4 MB | 76.7 – 78.5 µs |
| `neg` | 2048×4096 | 33.6 MB | 317.6 – 322.5 µs |
| `neg` | 4096×4096 | 67.1 MB | 639.9 – 644.8 µs |
| `neg` | 2048×16384 | 134.2 MB | 1307.1 – 1311.5 µs |

A straight-line fit `time = a·bytes + b` over all 61 full-machine runs (4–268 MB,
16 tensor shapes) gives **R² = 0.99995**, a slope of **102 GB/s**, and an intercept of
**b = −7.1 µs**. The intercept is slightly *negative* and is 9 % of the smallest kernel
measured, so there is no fixed start-up cost to charge.

All runs here use the full machine (32 cores). Fewer cores move the same bytes more
slowly — at one core the intercept reaches +245 µs — which is a separate effect,
derived in §10.

![neg kernel time is linear in HBM bytes through the origin, no fixed cost](cost_model_figures/s01_pointwise_baseline.png)

**Observation 2 — the arithmetic is free.** `gelu` and `exp` compute far more than `neg`
does (`exp` is a transcendental function), yet at the same tensor size all three take the
same time:

| tensor | traffic | `neg` | `gelu` | `exp` |
|---|---|---|---|---|
| 2048×1024 | 8.4 MB | 76.7 – 78.5 µs | 77.3 – 78.5 µs | 77.6 µs |
| 2048×16384 | 134.2 MB | 1307.1 – 1311.5 µs | 1314.5 µs | 1310.4 µs |

The spread across the three functions is under 1 %, which is the run-to-run spread of a
single one of them. So the arithmetic contributes nothing measurable: **the kernel time is
set entirely by moving bytes.**

**Model.** `time = bytes / BW`. No arithmetic term, no fixed start-up cost.

**How well understood is this term?** *Fully.* Both claims are direct readings of the
measurement, and the mechanism — a kernel that streams memory and does trivial work per
byte is limited by memory — needs no assumption about the hardware.

One thread is left open. The fitted rate is 102 GB/s, but a read-only kernel reaches about
150 GB/s. So `BW` is not one constant. §2 explains what it depends on.

### §2. The read/write ratio changes the effective bandwidth

**Observation — one bandwidth is not enough.** §1 measured 102 GB/s on a kernel that reads
one tensor and writes one. A kernel that almost only reads runs far faster per byte.

Two quantities are used from here on. **Effective bandwidth** is simply the traffic
divided by the measured time, `(R+W)/time`, where `R` and `W` are the bytes read and
written. The **write fraction** `w = W/(R+W)` says how one-sided the traffic is: 0 means
read-only, 1 means write-only, 0.5 means balanced.

| traffic mix | example op | write fraction `w` | effective bandwidth |
|---|---|---|---|
| almost all reads | `sumrow`, `read` (reduce to a tiny result) | 0.03 | **147 GB/s** |
| 2 reads : 1 write | `add` | 0.33 | **116 GB/s** |
| 1 read : 1 write | `neg`, `gelu`, `exp` | 0.50 | **106 GB/s** |
| almost all writes | `write` = `b[1,C] + c[R,1]`, both inputs tiny | 0.95 | **144 GB/s** |

These come from one sweep that varies only the traffic mix, at a single tensor height
(2048 rows) so tensor size cannot confound the comparison. A single `BW` is therefore
wrong: the rate is highest when traffic runs one way, and lowest when reads and writes
are balanced.

**Question.** What makes balanced traffic slower per byte than one-directional traffic?

**Hypothesis.** HBM is a shared bus that pays a **turnaround** cost when it switches
between reading and writing. The penalty falls on the overlap `min(R,W)`, which is 0 for
pure read or pure write and maximal at a balanced 1:1:

```
T = (R+W)/BW_peak + α·min(R,W)      ⇒   effBW = 1 / (1/BW_peak + α·f),  f = min(R,W)/(R+W)
```

This predicts a **symmetric valley**: the full rate at `w = 0` and `w = 1`, the minimum at
`w = 0.5`.

![§2 effective BW vs write fraction: a symmetric turnaround valley, read-only ≈ write-only](cost_model_figures/s02_read_write_ratio.png)

The two ends of the valley reach the **same height** — 147 GB/s reading, 144 GB/s writing.
Reads and writes therefore share one rate, so the model needs a single `BW_peak`, not a
separate read rate and write rate. The minimum sits at `w = 0.5`, exactly where `min(R,W)`
is largest.

**Model.** `time = (R+W)/BW_peak + α·min(R,W)`, with `BW_peak = 150 GB/s` and
`α = 0.00574 ns per byte`.

**How well understood is this term?** *Fully.* Bus turnaround is standard DRAM behaviour,
the predicted shape is symmetric and the measurement is symmetric, and the term vanishes
for one-directional traffic exactly as the mechanism requires. Nothing here is fitted
except the size of the penalty.

### §3. Reading a value another operation just wrote costs extra — a program-level condition

The two terms so far describe **one** operation. This section shows one thing they cannot
see, and states plainly that the model does not carry it.

**Observation.** Adding several tensors, `a + b + c + …`, is not a single instruction. It
compiles to a chain of two-input additions, each writing a result the next one reads. Every
one of those intermediate values is written to memory and read back, and the byte count
already includes all of it — so an `n`-way sum should cost exactly `n − 1` plain additions.
It costs more. At 2048 × 4096, where one addition takes 437 µs:

| sum | runs | predicted from bytes (µs) | measured (µs) | over |
|---|---:|---:|---:|---:|
| 3 inputs | 7 | 875 | 935 | +7 % |
| 4 inputs | 7 | 1312 | 1521 | +16 % |
| 5 inputs | 6 | 1749 | 2036 | +16 % |
| 6 inputs | 6 | 2186 | 2526 | +16 % |

**Question.** The extra bytes are already counted, so what is left?

**A control rules out the bytes.** Two *independent* additions, `(a + b)` and `(c + d)`,
move exactly the same bytes as the 3-input sum but have no dependency between them. They
measure 875 µs against a 875 µs prediction — **no margin at all**. The bytes are not the
problem; the dependency is.

**A second control rules out fusion.** Running the same dependent chain as three separate
kernels instead of one fused kernel gives 935 µs — the same as the fused version to within
1 µs. The penalty is present with or without fusion, so it belongs to the dependency, not to
the fused kernel.

**A third control identifies it.** Turning on the on-chip scratchpad keeps the
intermediate values in LX, so the chain no longer round-trips through HBM. The
same 3-input sum falls from 935 µs to **581 µs**, and the 6-input sum from 2526 µs to
**957 µs** — far below even the byte-count prediction, because most of the traffic has
stopped happening. Remove the round trip and the penalty goes with it.

**What it costs.** The excess grows with the length of the chain — about 0.14, 0.48, 0.66
and 0.78 extra additions' worth of time for one to four dependent reads — and the
separate-kernel version tracks it closely.

![One dependent sum measured four ways at 2048 by 4096: two independent additions sit exactly on the byte-count prediction, the dependent chain sits 7 % above it whether fused or split into separate kernels, and keeping the intermediate in LX drops it 34 % below](cost_model_figures/s03_read_after_write.png)

**Status — deliberately out of scope.** This is a cost *between* operations, and the model
in this report predicts *one* operation at a time. A single operation has no round-tripped
intermediate, so the term has nothing to attach to. It is recorded here because any future
model of a whole program must carry it, and because it is the same phenomenon as §18: a
value that does not fit in LX and therefore makes a trip to HBM. Every chained-sum
measurement is excluded from the accuracy figures in this report, and the exclusion is
stated wherever those figures appear.

### Part I data — every pointwise run

Predictions recomputed from the live model. Configurations measured more than once are
pooled to their median and the run count is shown, so a configuration measured many times
does not outvote one measured twice.

Over the whole population the model is **RMS 3.6 %, mean −1.5 %** across 89 runs, four of
them beyond ±10 %. Those four are not scattered: at the 32 cores the rate was calibrated
for, the model is **RMS 2.2 % over 85 runs with nothing beyond ±10 %**, and every one of
the four outliers was measured on a *fraction* of the machine — two at 8 cores, two at 16 —
all under-predicting by 13–15 %. The flat peak bandwidth is a full-machine number, and a
pointwise kernel evidently does not reach it on part of the machine, exactly as a reduction
does not (§6 needed an explicit core-count derate for the same reason). Four runs is far too
thin to fit one here, so it is recorded as a known residual; the deciding measurement is a
pointwise core-count ladder at fixed shape, which does not exist yet.

Two exclusions. Chained sums (§3) are multi-operation dependent chains, not single
operations, and their read-after-write cost is a program-level effect this model does not
carry. Adding a constant is excluded too: it lowers to an addition against a resident
constant, which makes it a broadcast operation, reported in §4.

| operation | n | RMS % | mean % | worst % | beyond ±10 % |
|---|---:|---:|---:|---:|---:|
| negate | 19 | 6.5 | -3.1 | -14.6 | 4 |
| add two tensors | 9 | 3.5 | -2.5 | -5.9 | 0 |
| apply a smooth activation | 3 | 1.9 | -0.3 | -2.6 | 0 |
| exponentiate | 3 | 2.2 | +0.3 | +3.1 | 0 |
| multiply two tensors | 3 | 1.7 | -1.1 | -2.3 | 0 |
| **all** | **37** | **5.0** | **-2.3** | **-14.6** | **4** |

A representative subset follows — full machine, ordinary shapes. All 37 configurations: `python3 tools/cost_model/part_tables.py 1 --full`.

‼ marks a run beyond ±10 %.

| operation | rows | columns | cores | runs | measured µs | predicted µs | err % |
|---|---|---|---|---|---|---|---|
| `add` | 2048 | 1024 | 32 | 11 | 108.3 | 108.0 | -0.3 |
| `add` | 2048 | 2048 | 32 | 1 | 215.0 | 215.9 | +0.4 |
| `add` | 2048 | 4096 | 32 | 14 | 436.4 | 431.8 | -1.0 |
| `add` | 8192 | 1024 | 32 | 2 | 451.4 | 431.8 | -4.3 |
| `exp` | 2048 | 1024 | 32 | 1 | 77.6 | 80.0 | +3.1 |
| `exp` | 2048 | 4096 | 32 | 1 | 319.7 | 320.0 | +0.1 |
| `exp` | 2048 | 16384 | 32 | 1 | 1310.4 | 1280.0 | -2.3 |
| `gelu` | 2048 | 1024 | 32 | 1 | 78.5 | 80.0 | +1.9 |
| `gelu` | 2048 | 4096 | 32 | 1 | 321.1 | 320.0 | -0.3 |
| `gelu` | 2048 | 16384 | 32 | 1 | 1314.5 | 1280.0 | -2.6 |
| `mul` | 2048 | 1024 | 32 | 1 | 107.4 | 108.0 | +0.5 |
| `mul` | 2048 | 4096 | 32 | 1 | 439.2 | 431.8 | -1.7 |
| `mul` | 2048 | 16384 | 32 | 1 | 1767.1 | 1727.4 | -2.3 |
| `neg` | 1024 | 4096 | 32 | 2 | 150.6 | 160.0 | +6.3 |
| `neg` | 2048 | 1024 | 32 | 2 | 78.4 | 80.0 | +2.0 |
| `neg` | 2048 | 16384 | 32 | 1 | 1311.5 | 1280.0 | -2.4 |
| `neg` | 4096 | 4096 | 32 | 2 | 642.4 | 640.0 | -0.4 |

---

## Part II — Other memory-bound ops

### §4. An operand small enough to stay resident raises the effective bandwidth

**Observation.** Several operations combine a full tensor with a much smaller one — a single
row, a single column, or a constant — reused across the whole output. The small operand is
read at its actual size, not expanded to match the output, so the traffic is one pass over
the large tensor and almost nothing else. That should make these run at the plain
read-then-write rate of §2, about 105 GB/s. They run faster: 115–123 GB/s at ordinary sizes,
for three additions and a multiplication alike, so it is the resident operand and not the
arithmetic that lifts the rate.

![Effective bandwidth against the number of columns, at 2048 rows, for the four operations with a small resident operand. All four run above a plain copy across the whole range, near 130 GB/s at the narrow end and settling toward 118 as the tensor widens. Adding a constant dips to 96 at 8192 columns](cost_model_figures/s04_broadcast_bandwidth.png)

**Question.** A single flat rate is a fair first cut, but it leaves errors up to 50 % once
the shape varies. How does the rate depend on shape?

**Two regimes, split at 1024 rows.** Above that the rate eases down with both dimensions, and
a two-term surface in `log₂(columns)` and `log₂(rows)` fits it to a few percent. Below it the
rate stops being monotonic and forms a **valley**, lowest exactly at `rows = columns / 64` —
that is, where the tensor is square measured in blocks rather than elements — and rising
steeply on the low side, gently on the high side. Fitting the two halves separately brings
the short-tensor error from 22 % to about 4.5 %.

![Effective bandwidth against row count below 1024 rows, one line per column count. The rate dips to a floor and recovers, with the floor sitting at rows = columns / 64 — 128 rows at 8192 columns, 256 at 16384](cost_model_figures/s04b_broadcast_small_rows.png)

The two row-broadcast operations run a few GB/s faster than the column and constant ones, so
each family carries its own constants; the shape is the same for both. Notably that shape is
*not* a property of the small operand — adding a constant, which has no operand to speak of,
collapses at short lengths exactly as broadcasting a row does.

**How well understood is this term?** *The direction is understood; the surface is fitted.*
That an operand small enough to stay resident is read once rather than streamed is plain, and
it correctly predicts the higher rate. The shape of that rate is not derived: it is a fitted
surface in two pieces with a boundary at 1024 rows, and the valley floor at
`rows = columns / 64` is a reproducible measurement with no established cause. The two
row-broadcast operations are essentially solved — 121 runs, none beyond ±10 % — and the
residual sits on the regime boundary.

### §5. The write operation builds a grid from two lines, and pays for the grid

**Observation.** `write` takes a row of values and a column of values and produces the full
grid of their sums — output element `(i, j)` is `column[i] + row[j]`. Its two inputs are
tiny: one holds as many values as the output has columns, the other as many as it has rows.
Almost all the traffic is the output. A model that charges for the bytes read and written
therefore expects it to run at the plain write rate, and for small tensors it does — about
135–145 GB/s. For a large tensor it collapses toward **40 GB/s**, and only when *both*
dimensions are large: making either one small restores the fast rate.

![Effective bandwidth against the number of columns, one line per row count, measured and modelled. The rate stays high whenever either dimension is small, and falls toward 40 GB/s only when both are large](cost_model_figures/s05_write_grid.png)

**Question.** The inputs are too small to explain any of this, so the cost must be in
producing the output rather than in fetching operands. What about producing it depends on
both dimensions at once?

**No mechanism established.** We do not have one. The candidate explanations all predict a
dependence on one dimension or on the output size alone, and the measured surface depends
on both together in a way none of them reproduces.

**Model.** Lacking a mechanism, the extra cost is charged as extra traffic, with a cap so
it cannot run away in the corner where both dimensions are largest (uncapped, that corner
over-predicted by 59 %):

```text
extra bytes = min( 2.0e-9 × rows^1.75 × columns^2.60 ,  2.4 × output bytes )
```

This takes `write` from 18.9 % error to **8.5 %** over 47 runs.

**How well understood is this term?** *Not at all — this is a black box.* Two exponents and
a cap were chosen to fit a surface whose cause is unknown. It is not derived from anything,
it will not extrapolate outside the measured rectangle, and its worst residuals remain large
(−30 % at 2048 × 8192). It is included because `write` is a real operation that would
otherwise be mispredicted by a factor of two, and it is deliberately isolated in its own
term so that nothing else in the model depends on it.

**Accuracy of §4 and §5 together**, counting every individual run (the Part II table at the end pools repeated runs of one configuration, so its counts are smaller). RMS **5.7 %**, mean −0.2 %, range −30.2…+17.0 %, over 281 points — 22 beyond
±10 %. Every measurement here is at 32 cores, so unlike Part I there is no core-count question
mixed in.

| op | n | RMS % | mean % | err range | >10 % |
|---|---:|---:|---:|---|---:|
| `bcast` | 59 | 3.0 | −0.1 | −8.9…+7.9 | 0 |
| `mulbcast` | 62 | 3.1 | +0.0 | −8.9…+8.9 | 0 |
| `copy` | 56 | 6.3 | −1.1 | −18.4…+7.8 | 7 |
| `bcastcol` | 57 | 6.6 | +1.7 | −18.5…+17.0 | 7 |
| `write` | 47 | 8.5 | −1.9 | −30.2…+11.6 | 8 |
| **all** | **281** | **5.7** | **−0.2** | **−30.2…+17.0** | **22** |

The two row-broadcast ops are essentially solved — 121 points between them, not one beyond ±10 %,
and mean error indistinguishable from zero. The residual is concentrated exactly where the
derivation said it would be: the `copy`/`bcastcol` regime boundary (7 points each, both directions)
and `write`, whose extra traffic is still an empirical power law rather than a mechanism, and which
owns the worst point in the section at −30 %. `write` is also the only op here whose error changes
sign with *both* R and C, which is why refitting it would produce another black box rather than an
explanation.

### §6. Reduction: read-bound, at a rate that falls with ROWS

**Observation.** A reduction — summing or taking the maximum along each row — reads a whole
tensor and writes almost nothing. Traffic is essentially all reads, so the turnaround cost of
§2 does not apply and the rate should simply be the read peak. It is not. The achieved rate
starts near the 150 GB/s peak on a short tensor and **falls as the tensor gets taller**,
settling around 113 GB/s: 149, 134, 121, 115 GB/s at 2048, 4096, 8192 and 16384 rows.

**Question.** Is that a size effect or a shape effect?

**Shape.** It tracks the number of *rows*, not the total size: holding the row count at 8192
and varying the width from 1024 to 4096 columns leaves the rate flat at 119–125 GB/s, even
though the tensor grows fourfold. All five reduction operations trace the same curve, so it
is not a property of the arithmetic being done.

**Model.** One curve in the row count:

```text
read bandwidth = min(150, 114 + 61 · exp(−rows / 3700))   GB/s
```

![Achieved read bandwidth against the number of rows, for five different reductions. All five follow the same declining curve, from about 150 GB/s on a short tensor to about 115 GB/s on a tall one](cost_model_figures/s06_reduction_read_rate.png)

**One exception.** Summing *down* the columns instead of along the rows walks memory
differently and shows no such falloff; it keeps a flat rate of about 113 GB/s. When that axis
is divided across cores each holds a partial result and they must be combined, but the
combine is small enough never to matter.

**A second effect: fewer cores reach less of the bus.** The curve above is the rate on the
full machine. A reduction is a streaming read over a shared bus, so with fewer cores active
fewer requests are in flight and less of the peak is realised. The derate is strongly
sub-linear — one core reaches 11 % of the bus, not the 3 % a proportional law would give:

| cores | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| fraction of full-machine rate | 0.11 | 0.22 | 0.43 | 0.54 | 0.54 | 1.00 |

Without it a reduction on few cores is mispredicted by up to −89 %. It is 1.0 on the full
machine, so nothing calibrated there moves.

**How well understood is this term?** *The read-bound part is understood; both rates are
fitted.* That a reduction reads a whole tensor and writes almost nothing is structural and
sets the shape of the cost correctly. Neither rate is derived. The core-count derate deserves
a sharper warning than that: it was calibrated largely on 1-, 2- and 4-core measurements that
the standing scope rules now exclude, so the only values the scored population still
exercises are 0.54 at 8 and 16 cores and 1.0 at 32 — most of the curve is no longer testable
against the data this report scores against. Every outlier in this section is the same shape
at 8 and 16 cores, over-predicted by about 20 %, which says the plateau is in the right place
and its value is wrong there. A repeated 8- and 16-core sweep across several shapes would
settle whether 0.54 is simply too low or whether the derate needs to depend on shape too.

### §7. Rearranging data costs what its access pattern costs, not what its bytes cost

**Observation.** Four operations move data without doing arithmetic: transposing a tensor,
swapping the two outer dimensions of a three-dimensional one, and joining two tensors along
either axis. Each compiles to a plain byte copy, so a byte model predicts one rate for all
four. Measured, two hold a flat rate at every shape — transposing at 116 GB/s, joining along
the columns at 106 — while the other two **fall as the row gets wider**, from about 110 GB/s
down to 44.

![Effective bandwidth against row width at a fixed 2048 rows. Transposing and joining along the columns stay flat; joining along the rows and swapping the outer dimensions fall as the row widens, with the model overlaid](cost_model_figures/s07_transport_access_pattern.png)

**Hypothesis.** What differs is how each walks memory. The hardware stores a row of `C`
values as `C/64` blocks, and building one output row means collecting its blocks from
scattered places. A wider row is more scattered fetches, so the cost should track the number
of blocks per row rather than the byte count.

**It does.** At a fixed byte count a wide operand is far slower than a tall one: joining
along the rows runs at 90 GB/s on a 2048 × 512 tensor and 44 on a 2048 × 32768 one. The two
flat operations are the two whose access stays contiguous.

**Model.** Each access pattern carries its own rate — flat for the two contiguous ones, and
for the other two a rate that declines with the number of blocks per row and, weakly, with
the operand length.

**One correction.** Swapping the outer dimensions has a sweet spot: sweeping the middle
dimension at a fixed shape, the rate peaks near 8 and falls off on both sides — 70, 100 and
72 GB/s at 2, 8 and 64. Eight blocks is the length of the contiguous run each core writes, so
below it the writes drop under a kilobyte and the rate falls about 13 GB/s per halving. That
penalty is modelled and takes those runs from a mean of −19.6 % to +0.1 %. The decline
*above* 8 is real but deliberately left alone: it is not separable from the row count (at
512 rows the large-middle points are accurate, at 2048 rows they reach −22 %) and it is
confounded with the compiler changing how it divides the work at large sizes. It is the
largest single contributor to this section's spread.

**How well understood is this term?** *The mechanism is identified; the rates are fitted.*
These operations move the same bytes and reach different rates, and the ordering follows how
scattered their access is in blocks — that much is mechanism. Each pattern then carries a
fitted rate rather than one derived from how scattered it is. The one modelled correction is
gated to the regime its data supports; the opposite end contradicts itself across shapes and
is left unmodelled.

### Part II data — every run behind the terms above

Predictions recomputed from the live model, not read from storage. Configurations measured more than once are pooled to their median and the run count is shown.

| operation | n | RMS % | mean % | worst % | beyond ±10 % |
|---|---:|---:|---:|---:|---:|
| broadcast a row across every row | 38 | 3.2 | -0.3 | -8.9 | 0 |
| multiply by a broadcast row | 38 | 3.3 | -0.2 | -8.9 | 0 |
| broadcast a column across every column | 36 | 6.6 | -0.1 | -18.5 | 5 |
| add a constant | 36 | 7.5 | -2.8 | -18.4 | 7 |
| join two tensors along the rows | 29 | 6.5 | +0.7 | +18.2 | 2 |
| transpose the outer dimensions | 27 | 6.0 | +1.8 | -12.0 | 2 |
| join two tensors along the columns | 23 | 5.7 | +0.8 | -21.8 | 1 |
| build a grid from a row and a column | 22 | 10.7 | -2.8 | -30.2 | 6 |
| transpose | 21 | 1.7 | -0.7 | -5.0 | 0 |
| sum along each row | 15 | 8.5 | +3.7 | +20.6 | 3 |
| read a tensor | 15 | 7.8 | +3.4 | +19.2 | 3 |
| maximum along each row | 13 | 8.8 | +3.4 | +19.8 | 3 |
| mean along each row | 13 | 8.8 | +3.7 | +19.8 | 3 |
| sum every element | 7 | 3.1 | +1.5 | +6.1 | 0 |
| sum down each column | 5 | 3.6 | +3.3 | +5.2 | 0 |
| **all** | **338** | **6.5** | **+0.3** | **-30.2** | **35** |

A representative subset follows — full machine, ordinary shapes. All 338 configurations: `python3 tools/cost_model/part_tables.py 2 --full`.

‼ marks a run beyond ±10 %.

| operation | rows | columns | cores | runs | measured µs | predicted µs | err % |
|---|---|---|---|---|---|---|---|
| `amax` | 2048 | 2048 | 32 | 2 | 57.9 | 58.0 | +0.2 |
| `amax` | 2048 | 8192 | 32 | 2 | 221.6 | 226.8 | +2.4 |
| `amax` | 4096 | 2048 | 32 | 2 | 129.7 | 129.0 | -0.5 |
| `amax` | 8192 | 1024 | 32 | 1 | 146.6 | 147.7 | +0.8 |
| `bcast` | 1024 | 2048 | 32 | 1 | 66.8 | 66.2 | -0.8 |
| `bcast` | 1024 | 8192 | 32 | 1 | 283.7 | 277.2 | -2.3 |
| `bcast` | 8192 | 2048 | 32 | 2 | 572.3 | 564.4 | -1.4 |
| `bcast` | 16384 | 2048 | 32 | 1 | 1165.2 | 1154.1 | -1.0 |
| `bcastcol` | 1024 | 2048 | 32 | 1 | 73.6 | 70.9 | -3.7 |
| `bcastcol` | 1024 | 8192 | 32 | 1 | 324.5 | 288.9 | -11.0 ‼ |
| `bcastcol` | 8192 | 2048 | 32 | 2 | 612.4 | 600.0 | -2.0 |
| `bcastcol` | 16384 | 2048 | 32 | 1 | 1251.9 | 1223.7 | -2.3 |
| `cat0` | 1024 | 1024 | 32 | 2 | 67.6 | 77.1 | +14.1 ‼ |
| `cat0` | 1024 | 32768 | 32 | 1 | 4428.5 | 4575.6 | +3.3 |
| `cat0` | 4096 | 1024 | 32 | 1 | 308.2 | 327.7 | +6.3 |
| `cat0` | 8192 | 2048 | 32 | 2 | 1700.9 | 1553.4 | -8.7 |
| `cat1` | 1024 | 1024 | 32 | 1 | 53.8 | 58.9 | +9.5 |
| `cat1` | 1024 | 4096 | 32 | 1 | 225.5 | 239.2 | +6.1 |
| `cat1` | 4096 | 4096 | 32 | 2 | 933.3 | 956.9 | +2.5 |
| `cat1` | 16384 | 2048 | 32 | 2 | 1964.5 | 1899.3 | -3.3 |
| `copy` | 1024 | 2048 | 32 | 1 | 83.0 | 69.8 | -15.9 ‼ |
| `copy` | 1024 | 8192 | 32 | 1 | 334.8 | 287.8 | -14.0 ‼ |
| `copy` | 8192 | 2048 | 32 | 3 | 605.1 | 590.7 | -2.4 |
| `copy` | 8192 | 4096 | 32 | 1 | 1221.6 | 1200.5 | -1.7 |
| `mean` | 2048 | 2048 | 32 | 2 | 58.9 | 58.0 | -1.5 |
| `mean` | 2048 | 8192 | 32 | 2 | 221.9 | 226.8 | +2.2 |
| `mean` | 4096 | 2048 | 32 | 2 | 128.7 | 129.0 | +0.2 |
| `mean` | 8192 | 1024 | 32 | 1 | 145.6 | 147.7 | +1.4 |
| `mulbcast` | 1024 | 2048 | 32 | 1 | 65.5 | 66.2 | +1.1 |
| `mulbcast` | 1024 | 8192 | 32 | 1 | 284.9 | 277.2 | -2.7 |
| `mulbcast` | 8192 | 2048 | 32 | 2 | 575.1 | 564.4 | -1.9 |
| `mulbcast` | 8192 | 4096 | 32 | 1 | 1173.1 | 1156.1 | -1.4 |
| `read` | 2048 | 1024 | 32 | 1 | 30.1 | 29.9 | -0.6 |
| `read` | 2048 | 4096 | 32 | 1 | 112.8 | 114.3 | +1.4 |
| `read` | 4096 | 2048 | 32 | 2 | 130.4 | 129.0 | -1.1 |
| `read` | 8192 | 2048 | 32 | 4 | 283.2 | 286.8 | +1.2 |
| `sumall` | 2048 | 2048 | 32 | 2 | 54.1 | 56.3 | +4.0 |
| `sumall` | 2048 | 8192 | 32 | 1 | 212.2 | 225.1 | +6.1 |
| `sumall` | 4096 | 2048 | 32 | 1 | 128.0 | 125.1 | -2.3 |
| `sumall` | 8192 | 1024 | 32 | 1 | 139.4 | 139.0 | -0.3 |
| `sumcol` | 2048 | 2048 | 32 | 2 | 70.6 | 74.3 | +5.2 |
| `sumcol` | 2048 | 8192 | 32 | 1 | 292.9 | 297.1 | +1.4 |
| `sumcol` | 4096 | 2048 | 32 | 1 | 145.1 | 148.5 | +2.3 |
| `sumcol` | 8192 | 2048 | 32 | 1 | 286.5 | 297.0 | +3.7 |
| `sumrow` | 2048 | 1024 | 32 | 1 | 30.6 | 29.9 | -2.4 |
| `sumrow` | 2048 | 4096 | 32 | 1 | 111.4 | 114.3 | +2.6 |
| `sumrow` | 4096 | 2048 | 32 | 2 | 128.2 | 129.0 | +0.6 |
| `sumrow` | 8192 | 2048 | 32 | 3 | 279.0 | 286.8 | +2.8 |
| `transpose` | 1024 | 1024 | 32 | 1 | 38.1 | 36.2 | -5.0 |
| `transpose` | 2048 | 16384 | 32 | 1 | 1156.5 | 1157.0 | +0.0 |
| `transpose` | 4096 | 1024 | 32 | 2 | 142.3 | 144.6 | +1.6 |
| `transpose` | 4096 | 4096 | 32 | 2 | 576.5 | 578.5 | +0.4 |
| `transpose_outer` | 1024 | 1024 | 32 | 2 | 316.0 | 332.9 | +5.3 |
| `transpose_outer` | 4096 | 1024 | 32 | 1 | 1310.6 | 1364.0 | +4.1 |
| `transpose_outer` | 8192 | 2048 | 32 | 11 | 5429.1 | 5938.8 | +9.4 |
| `transpose_outer` | 8192 | 4096 | 32 | 1 | 12783.8 | 12843.8 | +0.5 |
| `write` | 2048 | 1024 | 32 | 3 | 31.8 | 31.8 | +0.1 |
| `write` | 4096 | 8192 | 32 | 1 | 1130.1 | 872.3 | -22.8 ‼ |
| `write` | 8192 | 1024 | 32 | 3 | 134.7 | 131.2 | -2.6 |
| `write` | 16384 | 1024 | 32 | 1 | 284.6 | 271.0 | -4.8 |

---

## Part III — Matmul: memory *and* compute

A matrix multiply `A[M,K] @ B[K,N] → C[M,N]` is the first op here that can be **compute-bound**
rather than purely memory-bound, and the first with non-trivial dataflow across cores: it
performs `M·N·K` multiply-accumulate operations (MACs) on the systolic array. The planner tiles
the output into an `m × n` grid (it can also split the shared `K` into `k`, but strongly avoids
it), using `m·n·k` cores. Two quantities recur below: the **per-core tile** (`M/m` rows × `N/n`
columns each core computes) and the **HBM bytes read/written**, `R` and `W`.

### §8. Matmul takes two terms: the form of the model, and how it was fitted

**Observation.** For every op in Parts I–II, kernel time was fully accounted for by the bytes
moved. Matmul is different: its measured time is far larger than its HBM bytes would predict
under the copy model, and the excess grows with `M·N·K` — the MAC count — not with the byte
count. A `2048³` matmul moves the same order of bytes as a large copy but takes several times
longer. So matmul needs a second term, a **compute** term, on top of memory.

**Assumption.** We model matmul kernel time as a function of exactly two quantities: the compute
work (the MAC count) and the HBM memory traffic (`R`, `W`) — `T = f(compute, memory)`. Sections
§9–§12 pin down the form of `f`.

**Question.** How do the memory term and the compute term **combine** into one kernel time —
do they simply add, or does the accelerator do them at the same time?

**Strategy.** We build the model one term at a time, each in a regime where that term
dominates: first the memory term (on matmuls with almost no compute), then the compute rate (on
matmuls that are almost all compute), then how the two overlap. Only the compute rate (§10) is
truly isolated — it is fixed by a slope that does not depend on the other terms. The memory rate
(§9) and the overlap factor (§11) are correlated and were co-fit.

### §9. The memory term — measured on compute-free matmuls

**Isolation.** To measure the memory term alone, use matmuls where compute is negligible: a
very thin `K` (the output dominates → **write-heavy**) and a very thin `M` (large operands, tiny
output → **read-heavy**).

**Model.** In these compute-free corners the kernel runs at or below the §1 copy peak of
150 GB/s, and the read and write corners do not separate into distinct rates. So the memory term
is the §2 form with a single rate:

```
memory = (R + W) / 150 + α·min(R,W)    (α = 0.00574 ns/B)
```

![§9 baseline memory model vs measured on compute-free matmuls: within ~4% write-heavy, under-predicting the read-heavy large-N corner](cost_model_figures/s09_matmul_memory_term.png)

**Accuracy.** On the compute-free sweep (write-heavy `K ∈ {16,32}`, read-heavy `M ∈ {32,64}`,
`N` up to 4096) this baseline predicts the **write-heavy** corner to within ~4 %, but
**under-predicts the read-heavy corner — the large-`N`, thin-`M` shapes — by ~8–18 %** (figure):
there the read runs a little below the copy peak, plus a small fixed floor at tiny `M`. That
residual is minor next to the compute term that dominates real matmuls (§10), so it is carried
in the memory term rather than fit away. (The degenerate `K = 64` write-heavy point is dropped: a
single-stick contraction dim runs off-trend — `2048×64×2048` measures 56.6 µs, *below* both `K = 16`
and `K = 32`. It is a padding-free kernel — `K = 16` and `K = 32` pad the thin operand up to a full
64-element stick, adding preamble traffic — and it runs at ~158 GB/s, *above* the 150 GB/s copy peak
this term is built on, so it is not a clean memory-term point. And at fixed cores the §9 prediction is
independent of the `m·n` split: the byte count (`_fused_hbm_bytes`) carries no split dependence, and
the per-core tile area `M·N/(m·n)` is identical for every 32-core split, so the spill term is identical
too — `4×8`, `8×4`, `16×2` give the *identical* §9 prediction. Work-division dependence appears only
where the split changes the per-core geometry, in the compute and spill terms, §10 and §12, where the
split sweeps live.)

**How well understood is this term?** *Well.* It is the Part I bandwidth model applied
to a multiply's operands, measured on multiplies whose arithmetic is negligible so that the
memory cost is what remains. No new coefficient is introduced.

### §10. The arithmetic term — time per multiply-accumulate, divided by cores

**Observation.** With the memory term of §9 subtracted, the leftover time is arithmetic. The
simplest model is perfect parallelism: each core does an equal share at the same rate, so the
time should be linear in the number of multiply-accumulates and in one over the core count.

```text
arithmetic = multiply-accumulates / cores / peak
```

It holds. Doubling the cores halves the time, and doubling the summed dimension doubles it.
What matters is the *product* of the two division factors, not how the work is factored: the
two most balanced divisions of 32 cores land on the same time. A thin factor costs a few
percent, and a fully lopsided one about twice as much — that is §13's effect, not this one.

![Kernel time against one over the core count. Time halves when cores double. At equal cores the divisions collapse onto one line unless a factor is thin, where the lopsided cases sit about twice as high](cost_model_figures/s10_matmul_compute_rate.png)

**Fitting the rate without circularity.** The rate is the *slope* of time against one over
cores, and a slope is immune to any constant offset — including the overlap term of §11. So
unlike the memory rate, it is pinned without circularity. Two problem sizes swept from 4 to 32
cores give straight lines whose slopes differ by exactly the factor their arithmetic does. The
fits carry a positive intercept of about 100 µs, which is the memory floor, and is exactly why
the rate is read off the slope rather than the intercept.

**How well understood is this term?** *The form is understood; the rate is measured, not
derived.* The proportionality is clean and directly measured. The sustained rate of 1140 per
nanosecond per core is fitted on the runs where arithmetic dominates; it sits below the
array's nominal peak and nothing here explains the gap. Its absolute level is mildly
correlated with the overlap fraction of §11, so it is quoted as a small range rather than a
spuriously exact number.

### §11. Arithmetic and memory partly overlap — a fitted fraction

**Observation.** §9 and §10 give a multiply two costs: moving its operands, and doing its
arithmetic. Adding them over-predicts. A real multiply is faster than the sum, by more as
the two costs approach each other in size.

**Hypothesis.** The array computes on one block of operands while the next streams in, so
the two run partly at the same time. At best the shorter of the two disappears entirely
inside the longer, which caps the saving at `min(arithmetic, memory)`.

**Model.** Hide a fixed fraction of that cap:

```text
time = arithmetic + memory − γ · min(arithmetic, memory)         γ = 0.46
```

`γ = 0` is fully serial; `γ = 1` is a perfectly pipelined array. Equivalently this is
`max(arithmetic, memory) + (1 − γ) · min(...)` — the slower stream plus whatever fails to
hide.

**What was tested.** Perfect overlap is refuted outright: plain `max` scores 17 / 26 / 38 %
on three progressively wider sets of runs, against 8.7 / 14.3 / 29.0 % for the form above.
Seven alternative shapes for the leftover were then fitted and scored — a fixed start-up
cost, a charge on stores only, one tile's traffic, a softened crossover, a shared-port
bound, a balance-dependent fraction, and a core-count-dependent one. On a genuine hold-out
of 239 runs never used for fitting, **none beats the shipped form** (16.19 %; the closest
alternative is 16.20 %, the rest trail to 17.6 %). The *shape* of this term is as well
supported as the data can make it.

**The coefficient is not.** Different populations want different values, in opposite
directions, when the spill coefficients of §12 are re-optimised at each:

| γ | plain multiply | batched | coarse-tiled | all |
|---|---:|---:|---:|---:|
| 0.30 | 15.65 | **40.81** | **23.54** | **28.57** |
| **0.46 (shipped)** | 14.25 | 41.88 | 24.73 | 28.93 |
| 0.60 | **13.56** | 43.12 | 26.51 | 29.67 |

Plain multiplies want more overlap; batched and coarse-tiled ones want less, and they
outnumber the plain ones. So 0.46 is a compromise, not a measured constant. It is also
entangled with §12: freeing either coefficient alone gains nothing out of sample, freeing
both together gains about half a point, with γ landing at 0.58–0.70 every time.

![Prediction error against the share of time spent moving memory, for every 32-core multiply. Open markers are the model without this term, filled with it. Without it the error grows steadily as memory dominates; with it the realistic runs stay inside ±10 %](cost_model_figures/s11_compute_memory_overlap.png)

**How well understood is this term?** *This is a fit, not a mechanism.* The form is
justified — arithmetic and transfer genuinely do overlap, and every alternative shape tested
does worse out of sample. But nothing derives the value 0.46, and the honest reading is that
it is where two disagreeing populations balance. The two dissenting groups both carry known
unmodelled effects of their own, so neither is a fair arbiter yet; the value should be
revisited once those are fixed, not tuned against them now. An earlier expectation that the
fraction should fall as cores rise was tested directly and **is not supported**: with the
arithmetic rate pinned first on the one- and two-core runs, the fitted fraction goes
0.20, 0.60, 0.84, 0.60, 0.56, 0.52 from 1 to 32 cores — no trend.

### §12. A per-core tile too large for the accumulator forces the operands to be re-streamed

**Observation.** With the memory, arithmetic and overlap terms in place, balanced matmuls on
many cores still leave a residual, and it **grows once each core's output tile passes about
128 KB**. Below that the residual sits near zero; above it, it climbs steeply, reaching +277 µs
at the largest tile measured.

![Residual against per-core output tile size. Near zero while the tile is small, climbing once it passes about 128 KB](cost_model_figures/s12_accumulator_spill.png)

**What overflows is not LX.** The obvious reading is the scratchpad, and it is
wrong: the compiled programs show a matmul allocating **none** of it, on all 316 runs. The
resource that overflows is on the compute side — the accumulator the array holds the running
output in. When the tile no longer fits, both operands must be streamed again for reuse.

**Model.** Charge that re-stream as extra read traffic, with the re-read fraction growing as
the tile passes the threshold:

```text
spill = (operand A bytes + operand B bytes) · min(1.5, max(0, 0.45 · log₂(area / 65536)))
```

**How well understood is this term?** *The cause is identified, the shape is fitted.* The
threshold is measured and its location is consistent across the sweep, and the section
establishes what it is *not* — the scratchpad — which is worth more than the fitted curve. The
0.45 slope and the 1.5 ceiling are chosen to fit; because the trigger is compute-side,
charging the re-stream to memory rather than to arithmetic is partly arbitrary and the data
do not separate the two. This term is also entangled with the overlap fraction of §11:
freeing both together gains about half a point out of sample, more than either alone, so
neither is independently pinned. Past the ceiling the largest balanced tiles are under-predicted
by about 15 %.

### §13. Splitting one output dimension many ways makes cores re-read a shared operand

**Observation.** With the four matmul terms so far in place, a *lopsided* division of work
still under-predicts badly. On 32 cores at a fixed problem, dividing the output `8 × 4` is
accurate; `16 × 2` misses by 40 %, and `32 × 1` by 61 %. Each core's tile has the same *area*
in every case, so the spill term of §12 sees no difference — what matters is the *shape* of
the division, not the size of the tile.

![Residual against per-core tile size. A balanced division stays near zero at any tile size; dividing either output dimension many ways climbs once the tile passes about 256 KB, and the two-sided term (dashed) tracks both](cost_model_figures/s13_split_shape.png)

**Hypothesis.** Cores that split the same output dimension need the same operand. The cores
sharing the row dimension all want the same weight columns; those sharing the column
dimension all want the same activation rows. Past a group of about eight, that shared
operand is fetched again from HBM rather than broadcast once. This should bite only
when the tile is large *and* a dimension is divided many ways — and it should be
**asymmetric**, because slicing the *longer* output dimension thinly hurts more than slicing
the shorter one. It does: at 32 cores on the long dimension the cost is about twice the same
fanout on the short one.

**Model.** Two additive terms, one per output dimension, charged after the overlap of §11:

```text
split = c_L · max(0, area − a₀) · max(0, log₂(long fanout / 8))
      + c_S · max(0, area − a₀) · max(0, log₂(short fanout / 16))
area = per-core tile;  a₀ = 131072 elements (≈ 256 KB);  c_L = 2.6e−3, c_S = 2.9e−3 µs per element
```

The threshold of eight on the longer dimension is the compiler's own group limit; the
sixteen on the shorter one is fitted. Both terms are exactly zero for a balanced division
and for small tiles, so nothing in §9–§12 changes. On the lopsided runs the error falls from
**36 % to 15 %**, and the mean bias from −29 % to −2 %. The deepest case — splitting a very
long dimension 32 ways — is improved but still 24 % under.

**How well understood is this term?** *Partly.* That a lopsided division wastes cores is
clear, and the term fires only where the division is lopsided. Which geometry is penalised,
and by how much, is fitted. One residual sits alongside it and survives a noise check: error
is markedly worse when each core gets very few output columns — 18.5 % below 128 columns per
core against 8.3 % between 128 and 512. Sixty-four columns is one hardware block, the
narrowest tile a core can be given, so array underfill is the natural reading.

### §14. A batched multiply is set by the operands' memory layout, not their bytes

**Observation.** A batched multiply performs the same multiply once per batch, and the
compiler never spreads a batch across cores — it keeps each one on the same cores and
iterates — so the cost should be the number of batches times the cost of one multiply. It is
not. Past four batches the measured time settles at a shape-dependent **2.1–4.7×** that
prediction and stays there as the batch grows, while a plain unbatched multiply of the same
per-batch shape is predicted correctly.

The byte count already charges for every batch, and so does the arithmetic count, so the gap
is a *rate* rather than a missing quantity. It is not extra bytes either. A tensor can be
arranged in device memory in more than one order; running one batched multiply under all four
combinations of its two operands' orders — **the same bytes every time**, with the compiled
program confirming no copy is inserted:

| operand A | operand B | time (µs) | relative |
|---|---|---:|---:|
| compiler default | compiler default | 1847 | **3.32×** |
| compiler default | alternative | 1293 | 2.33× |
| alternative | compiler default | 1062 | 1.91× |
| alternative | alternative | 556 | 1.00× |

Same work, same bytes, 3.3× the time. Over 138 runs and 17 shapes the default pair is
**2.82×** the alternative.

![Each batched matrix multiply timed under all four combinations of its two operands' memory layouts, divided by its own fastest layout. 138 runs over 17 shapes. The compiler default is 2.82× slower, with the two single-operand swaps in between, at identical byte counts](cost_model_figures/s14_bmm_operand_layout.png)

**Slower memory or slower arithmetic?** Arithmetic. Time per unit of arithmetic is **flat at
about 215 µs per billion multiply-accumulates** across the whole default-layout group, 16
shapes over a 16-fold range. A memory effect would track each shape's ratio of bytes to
arithmetic; it does not. The default order walks the batch outermost, so consecutive batches
share no operand locality and the array is starved.

**Only the default order is modelled.** The faster arrangement is reachable only by opting into the alternative operand
layout, which is not the default — so every batched multiply the compiler
emits today has both operands in the default order. The model therefore charges one measured
rate for that case, **160 MAC/ns/core**, and prices nothing else. An earlier version fitted
all four combinations with three constants in an additive form; it reproduced them well, but
it spent that complexity on configurations nothing emits, and it has been removed. Runs that
forced another arrangement are excluded from scoring for the same reason. The measurement
above stays as evidence that the layout is worth roughly 2.8×, which is what a planner would
need to know if it ever chose the order.

| batched matmul, default layout | runs | RMS % | mean % | beyond ±10 % |
|---|---:|---:|---:|---:|
| `bmm_wd` | 64 | 41.7 | −25.7 | 35 |
| `bmm_wd_3d2d` | 37 | 15.7 | −5.7 | 5 |
| `bmm_layout` (both-default rows) | 34 | 28.0 | −11.7 | 9 |
| **all** | **135** | **33.0** | **−16.7** | **49** |

**How well understood is this term?** *The cause is established; the rate is fitted.* That
layout rather than byte count sets the rate is settled by a controlled experiment — 2.82× at
byte-identical traffic with no copy inserted, and a flat cost per unit of arithmetic that
rules out a memory explanation. The rate itself is fitted at one work division and 32 cores,
so it is confounded with that geometry. Three regimes keep the plain rate because their data
cannot support one: fewer than 8 cores, a batch of 2, and the shared-weight variant, where a
flat rate is refuted outright. The group's 33.0 % is the largest in this Part, and `bmm_wd`
at 41.7 % is the bulk of it — 101 of those runs predate layout recording and cannot be
checked against this term at all.

### Part III data — matmul

Over the whole scored population of divided matmuls the model is **RMS 15.1 %, mean −0.3 %,
range −48…+48 %**. That is the honest headline, and matmul is the weakest Part: the mean says
the model is unbiased, the spread says it is not precise, and the error is two-sided so no
single scaling fixes it.

Splitting the power-of-2 shapes by regime shows where the spread lives:

| regime | runs | RMS % | mean % | range % |
|---|---:|---:|---:|---|
| balanced division, ordinary shape | 167 | 14.4 | −1.1 | −48…+35 |
| divided more than 8 ways | 50 | 15.5 | −4.0 | −45…+46 |
| very short summed dimension (K ≤ 128) | 17 | 14.1 | +2.8 | −28…+33 |
| small output (either side ≤ 512) | 19 | 27.0 | +22.4 | −8…+48 |
| **all** | **253** | **15.9** | **+0.4** | **−48…+48** |

The split term of §13 does its job — a lopsided division is no longer distinguishable from an
ordinary one. That leaves the *ordinary* case at 14.4 %, which is the real weakness: the
spread runs through the regime the model is built for, not through an exotic corner that
could be gated away. The one distinct group is a small output, over-predicted by 22 %, where
the per-core tile underfills the array and no term accounts for it.

Predictions recomputed from the live model, not read from storage. Configurations measured more than once are pooled to their median and the run count is shown.

| operation | n | RMS % | mean % | worst % | beyond ±10 % |
|---|---:|---:|---:|---:|---:|
| plain matrix multiply, work division forced | 177 | 17.1 | +0.1 | +48.2 | 64 |
| batched multiply, operand layouts varied | 93 | 19.2 | -7.0 | -68.4 | 22 |
| batched multiply, a weight matrix per batch | 40 | 41.8 | -26.7 | -92.2 | 22 |
| batched multiply, one shared weight matrix | 24 | 19.3 | -8.4 | -53.0 | 5 |
| **all** | **334** | **22.2** | **-5.7** | **-92.2** | **113** |

A representative subset follows — full machine, ordinary shapes. All 264 configurations: `python3 tools/cost_model/part_tables.py 3 --full`.

‼ marks a run beyond ±10 %.

| operation | M | K | N | batches | layouts | division | cores | runs | measured µs | predicted µs | err % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bmm_layout` | 1024 | 1024 | - | 2 | default/default | - | 32 | 1 | 253.1 | 139.7 | -44.8 ‼ |
| `bmm_layout` | 1024 | 1024 | - | 8 | fast/fast | - | 32 | 1 | 806.8 | 654.9 | -18.8 ‼ |
| `bmm_layout` | 1024 | 2048 | - | 4 | default/fast | - | 32 | 3 | 1296.5 | 1319.4 | +1.8 |
| `bmm_layout` | 2048 | 2048 | - | 2 | fast/default | - | 32 | 2 | 616.8 | 399.0 | -35.3 ‼ |
| `bmm_wd` | 1024 | 2048 | - | 1 | - | - | 32 | 1 | 1846.2 | 1838.6 | -0.4 |
| `bmm_wd` | 1024 | 2048 | - | 4 | - | - | 32 | 4 | 2207.8 | 1838.6 | -16.7 ‼ |
| `bmm_wd` | 1024 | 2048 | - | 16 | - | - | 32 | 1 | 7356.3 | 7354.5 | -0.0 |
| `bmm_wd` | 2048 | 2048 | - | 4 | - | - | 32 | 4 | 4383.3 | 3616.9 | -17.5 ‼ |
| `bmm_wd_3d2d` | 1024 | 2048 | - | 2 | - | - | 32 | 2 | 279.0 | 263.8 | -5.5 |
| `bmm_wd_3d2d` | 1024 | 2048 | - | 8 | - | - | 32 | 2 | 1394.3 | 1390.6 | -0.3 |
| `bmm_wd_3d2d` | 2048 | 2048 | - | 2 | - | - | 32 | 2 | 549.6 | 512.5 | -6.8 |
| `bmm_wd_3d2d` | 2048 | 2048 | - | 8 | - | - | 32 | 2 | 2723.5 | 2766.0 | +1.6 |
| `mmwd` | 1024 | 1024 | 1024 | 1 | - | 4×8 | 32 | 1 | 67.4 | 69.9 | +3.6 |
| `mmwd` | 1024 | 4096 | 1024 | 1 | - | 4×8 | 32 | 3 | 191.4 | 201.4 | +5.3 |
| `mmwd` | 2048 | 4096 | - | 1 | - | - | 32 | 10 | 1322.2 | 1398.5 | +5.8 |
| `mmwd` | 4096 | 2048 | 4096 | 1 | - | 8×4 | 32 | 2 | 1706.4 | 1450.6 | -15.0 ‼ |

---

## Part IV — Coarse tiling: when one operation becomes a loop

**Coarse tiling** splits one large operation into a sequence of smaller ones and runs them
in a loop. The compiler does this so that each step works on a piece small enough to keep
in LX instead of going back to HBM. Two examples run here: `softmax(x)`, which is a chain of five steps fused into one
loop, and a matrix multiply whose rows are processed a block at a time.

Tiling changes something the earlier Parts never had to consider: **the same tensor can be
transferred more than once.** Parts I–III could count each tensor once because each kernel
touched it once. Inside a loop that is no longer true, and getting the count right is most
of this Part.

### §15. Tiling changes how many times each tensor moves

**Observation — traffic grows with the number of tiles, while the arithmetic does not.**
Take one matrix multiply, `4096×2048 @ 2048×2048`, and run it at several tile counts. The
multiply performs exactly the same arithmetic every time. The bytes transferred do not:

| tiles | runs | traffic (MiB) | mean measured time (µs) |
|---:|---:|---:|---:|
| 1 | 10 | 40.0 | 839.3 |
| 2 | 7 | 48.0 | 781.9 |
| 4 | 10 | 64.0 | 678.9 |
| 8 | 12 | 96.0 | 873.2 |
| 16 | 8 | 160.0 | 1310.1 |

(Traffic is what the current extraction charges; an older extraction of the same runs
recorded no repeats at all and is excluded from scoring for that reason.) Traffic
quadruples. Time first falls, then rises. Charging each tensor once reproduces the
**fall** — smaller pieces spill less, which §18 covers — but nothing in a once-counted
byte model can produce the **rise**.

**Question.** Which tensor is moving repeatedly, and how many times?

**Hypothesis.** A loop re-runs its body once per tile. If a tensor's address varies along
the dimension being tiled, each pass touches a different slice and the whole tensor is
transferred once. If its address does not vary along that dimension, every pass re-reads
the same bytes.

Two pieces of the compiled program are needed to decide this, and neither is sufficient
alone.

*Which dimension is tiled* comes from metadata the tiling pass attaches to each operation:
a trip count, and the position of the tiled dimension. In the dumped program for this
multiply at four tiles, that reads a trip count of 4, tiled dimension 0 (the rows), and the
row extent shrinking from 4096 to 1024 — one tile's worth.

*Which operands vary along it* comes from the addressing. Writing `i0` for the row index
and `r0_0` for the summed index:

```text
A  at  r0_0 + K·i0     contains i0  -> varies along the tiled dimension
B  at  i1   + N·r0_0   no i0        -> identical bytes on every pass
```

The addressing on its own proves nothing: it is character-for-character the same before
and after tiling. It is the metadata that says dimension 0 is now traversed in 4 blocks;
the addressing then says `A` follows those blocks and `B` does not.

**Model.** Each tensor argument gets a **move count** — how many times the loop transfers
it:

```text
moves(tensor) = product over loop levels L of
                  ( trip count of L   if the address does not vary along L's tiled dimension
                    1                 otherwise )
```

Traffic is the sum of `size × moves` over every tensor, except that an input shared by
several operations of one fused loop is counted once rather than per consumer.

Four cases occur in the measurements:

| what the loop tiles | output | operand A | operand B |
|---|---:|---:|---:|
| rows | 1 | 1 | **tiles** |
| the summed dimension | **tiles** | 1 | 1 |
| both (nested) | **inner tiles** | 1 | **outer tiles** |
| helper steps a tiled reduction adds (below) | — | — | — |

The second row is worth reading twice: when the loop splits the dimension being summed
over, both operands advance, but the *output* is revisited every pass, because each pass
adds its partial result into the same place.

**The fourth case is not a footnote.** Tiling a reduction makes the compiler insert extra
steps — initialising the running result, accumulating each tile into it, sometimes copying it
out. These are separate operations with their own tensors, and they carry **30 %** of a
summed-dimension multiply's traffic and **37 %** of a nested one's at eight tiles, rising to
about half at two. They obey the same rule, but they are why the loop, not the operation, is
the unit that has to be costed.

**How well understood is this term?** *The rule is derived, its application is not yet
settled.* The move counts are read from the compiled loop rather than fitted, and the
nested case is the one a single number per operation cannot express. But three successive
readings of the same loops produced three different answers for the helper steps above,
differing by up to 2.6× in charged traffic, so those counts are only as good as the current
reading. Only the first row of the table currently has tiled measurements that survive the
data checks; rows two and three rest on the rule plus untiled anchors, and their
re-measurement is still outstanding.

### §16. What a repeated read actually costs

§15 established *which* tensor repeats. This section asks what one repeat costs.

**Observation.** Hold the multiply's shape fixed and add tiles. Each extra tile is one
extra pass over operand `B`, so the cost of a pass can be estimated from the difference
between two tile counts. Using `(time at 16 tiles − time at 8 tiles) / 8` over every
in-scope run at 4096 rows and 2048 summed elements, widening the output to vary the size of
`B`:

| cores | size of `B` (MiB) | runs at 8 / 16 tiles | cost of one extra pass (µs) | one full pass at 150 GB/s (µs) |
|---:|---:|---:|---:|---:|
| 32 | 2 | 2 / 2 | 17.1 | 14.0 |
| 32 | 4 | 3 / 3 | 27.4 | 28.0 |
| 32 | 8 | 4 / 4 | 54.6 | 55.9 |
| 32 | 16 | 3 / 3 | 100.3 | 111.8 |
| 16 | 2 | 2 / 2 | 16.4 | 14.0 |
| 16 | 4 | 2 / 2 | 31.1 | 28.0 |
| 16 | 8 | 2 / 2 | 18.3 | 55.9 |
| 16 | 16 | 2 / 2 | −3.4 | 111.8 |

**What this does and does not show.** On the full machine the cost tracks the size of `B`
closely enough to rule out a fixed per-tile overhead, which would be constant down the
column where these span 5.9×. That is what this section rests on. The rest of the table is a
warning: the 16-core rows do not scale at all — they rise, fall and go negative, so the
estimator is unstable there — a second, equally reasonable estimator contradicts the 32-core
result at two of four sizes, the fitted exponent is 0.865 rather than the 1.0 strict
proportionality needs, and three of the eight rows exceed a full pass, which a transfer
cannot do. The *existence* of a size-dependent transfer is established; its coefficient is
not.

**Is the repeat served from LX instead?** No, and the compiler settles this
without appeal to timing. On-chip residency for an input is granted by copying it in once
and reusing the copy, which the compiler grants only when the tensor is read more than
once. Its bookkeeping is per operation, and the whole loop is one operation, so it records
a single use, declines the copy, and every pass goes to HBM. Across the coarse
matmul runs, **every** operand read is marked as coming from HBM and **none** from
LX — 1180 reads over all six coarse operations, 266 of them in scope, zero
on-chip under every filter tried. The same accounting gives a fused softmax on-chip
placement for two-thirds of its inputs, so the mechanism works; it does not reach this case.

**Model.** A repeated read is charged as memory traffic at the Part I rate, scaled by a
constant:

```text
extra traffic = size(tensor) × (moves − 1) × reread_scale        reread_scale = 0.85
```

charged at the peak rate, and **not** subject to the per-tile derates of §17 — those
describe a short stream, whereas this is one long pass over a whole operand.

**How well understood is this term?** *The effect is established, the coefficient is
fitted.* That the operand is re-read follows from two independent places — the loop
structure of §15, and the compiler's own residency decision above. The value 0.85 does not:
it is chosen to fit, it sits below 1.0 without a reason that predicts *how far* below, and
it acts on the same runs as the underfill term of §17, so the present data does not divide
the two.

### §17. Underfill: a short per-core tile runs the pipeline below peak — the `eff` term

**Observation.** Setting aside the spilling case of §18, a coarse-tiled kernel's speed still
depends on the **per-core tile height** — the rows each core streams per tile,
`h = ROWS / (cores · tiles)`: with too few rows the **effective bandwidth drops**. Sweeping the
tile count on softmax over the scored population — 32 cores, at least 1024 columns, and only the
runs whose working set fits in LX — the effective bandwidth climbs from ~48 GB/s at a 2-row tile
(`h = 2`) to a ~150 GB/s plateau by `h ≈ 16`, then mildly declines (figure).

![§17 the coarse underfill: softmax effective BW climbs with the per-core tile height, plateaus at h≈16](cost_model_figures/s17_underfill.png)

**Model (calibrated).** A pipeline-fill efficiency `eff ≤ 1` multiplies the memory term, keyed on
the per-core tile height `h = ROWS / (cores · tiles)`:

```
eff = min(0.95,  (h / 13)^0.68)          memory term = (R + W) / BW_eff / eff
```

It plateaus at 0.95 by `h ≈ 16` and derates below (≈0.45 at `h = 4`, ≈0.28 at `h = 2`). A
cross-`COLS` control (same `h`, double the tile bytes → same per-byte cost) confirmed it keys
on **rows (`h`), not tile bytes**. **On the softmax regime where the intermediates fit LX, this
gives RMS 5.9 %** (mean −1.2 %, over 45 points) — the coarse-tiling model is accurate once §18's
spill is set aside.

**Two residuals, both left unmodeled.** (1) Above `h ≈ 32` the efficiency mildly declines
(150 → 131 GB/s) while the model holds the 0.95 cap — a small, rows-driven droop. (2) A **tiled
matmul** (`matmul_row_tiling`) appears to underfill on *compute* the way softmax underfills on
memory — beyond a few tiles its time climbs as each tile gets fewer rows — but the available data
is thin, non-current, and partly non-monotonic, so it is **flagged, not modeled** (tiled matmuls
take `pt_eff = 1`; a clean tile-count sweep is queued).

**How well understood is this term?** *The cause is clear, the curve is fitted.* A
pipeline that is handed too few rows cannot fill, and the measured rate climbs and then
plateaus exactly as that predicts. The three constants describing the climb are fitted on
reductions and reused for multiplies without independent calibration.

### §18. A tile too large for LX moves at a lower rate — the `s_lx` term

**Observation.** §17 explained why a *short* tile is slow. A *tall* tile is slow too, for an
unrelated reason. Grouping the tiled softmax runs on the full machine by the working set
each core holds — about `2 × (rows per core) × columns × 2 bytes` — the achieved bandwidth
falls steadily as that working set grows:

| per-core working set (MB) | runs | achieved bandwidth (GB/s) |
|---|---:|---:|
| 0.25 – 0.5 | 18 | 99.4 |
| 0.5 – 1 | 9 | 96.2 |
| 1 – 2 | 8 | 89.7 |
| above 2 | 8 | 78.2 |

**Question.** Are those bytes being missed by the byte count, or are the same bytes simply
moving more slowly?

**They are moving more slowly.** At the largest working set the byte count already marks
those intermediates as memory traffic and includes them — 604 MB of reads and writes for
the tallest tile measured. If bytes were missing, the predicted *traffic* would be too low;
what is wrong instead is the predicted *rate*.

![Achieved bandwidth against the working set each core holds, for tiled softmax on the full machine. The rate is near 100 GB/s for a small tile and falls steadily as the tile grows. The dashed line is the fitted threshold, not a documented hardware capacity](cost_model_figures/s18_lx_spill.png)

**Model.** Keep the bytes and lower the rate:

```text
bandwidth ×= (cap / working set) ^ 0.15          when working set > cap
```

with `cap = 512 KB` for a tiled reduction and `cap = 2 MB` for a tiled matrix multiply. It
is inert below the cap, so it touches only the tall-tile end. On tiled softmax at 32 cores
it takes the error from **10.7 % to 6.0 %** over 60 runs, and the worst point from −39.8 %
to −17.8 %.

**How well understood is this term?** *The direction is right; the thresholds are fitted
and one of them is unexplained.* That a tile too large for LX costs extra is not
in question, and the error grows with working set exactly as that predicts. Three things are
weaker than they look:

*The 512 KB threshold does not match the hardware.* This repository documents 2 MB of
scratchpad per core, of which the compiler can allocate about 1.6 MB. The reduction cap of
512 KB is roughly three times smaller than any documented capacity, and is not derived from
one — an earlier draft of this section claimed it matched the usable capacity, which was
simply the fitted value restated.

*The second cap is barely identified.* Sweeping the multiply cap over the 19 runs it acts
on, everything from 1 MB to infinity lies within 0.44 percentage points of error. 2 MB is
the best value by that measure, but 512 KB is not excluded, and it gives the smaller mean
bias. So the honest statement is not that a multiply *needs* a different capacity — it is
that the data do not determine one. The 2 MB value is at least the documented per-core
capacity; the 512 KB one is not.

*The population is narrow.* The table is softmax on 32 cores; widening it to other tiled
reductions or fewer cores changes both the levels and the trend, so the curve is calibrated
on one operation at one core count and applied beyond it.

### §19. A fused chain has a floor set by how many elements it touches, not how many bytes it moves

**Observation.** Every term so far computes time as bytes divided by a rate: `moves`
multiplies the byte count, `eff` and `s_lx` change the rate it is divided by. Predicted time
is therefore proportional to the bytes counted. On a fused chain — a softmax, where several
stages run as one kernel — that proportionality breaks. Fix the shape at 4096 × 2048 and run
it on **one** core, varying only how finely it is tiled:

| tiles | HBM bytes the model counts (MB) | predicted from those bytes (µs) | measured (µs) |
|---:|---:|---:|---:|
| 1 | 117.4 | 1072 | 5418 |
| 4 | 54.5 | 810 | 5507 |
| 8 | 44.0 | 595 | 5388 |
| 16 | 33.6 | 415 | 4576 |

Two things are wrong, and the second is the important one.

The counted traffic falls **3.5×** — tiling moves the intermediates into LX, so they stop
being charged as HBM traffic, which is correct accounting — and the prediction duly falls
with it. The measurement does not.

But the prediction is already **5× too fast at one tile**, before tiling enters the picture:
1072 µs against 5418 measured. A byte-over-bandwidth term assumes the bus is the limit. On a
single core it is not — one core cannot saturate HBM — so the kernel is bound by how fast
that core passes elements through the chain, and reducing HBM traffic cannot speed up
something that was never waiting on HBM. Across the four tile counts the byte-only model
under-predicts by **86 %** on average.

**Question.** If the cost is not the bytes, what is it?

**Hypothesis.** Tiling changes where the intermediate values live, but not how many values
there are: every stage of the chain still has to touch every element. What shrinks is the
accounting, not the work. If the chain is limited by how fast a core can pass elements
through it, the cost should follow the element count and the core count, and ignore tiling.

**What the measurements do and do not rule out.** Three facts constrain the answer:

*The element count does not move.* In the table above the largest operand holds 8.4 million
elements at every tile count — it stays fixed while the bytes fall and the time stays flat.

*The shortfall scales with cores, not with size.* The gap between measurement and a
byte-only model shrinks roughly in proportion to `1/cores`, the signature of a per-core rate
rather than a fixed per-kernel cost.

*Cores still matter when the working set is held fixed.* The spill term of §18 depends on
`cores × tiles`, so that product can be held constant while cores varies. At 2048 × 512:
`(1 core, 8 tiles)` takes 591 µs, `(2, 4)` takes 296 µs, `(8, 1)` takes 89 µs. Time falls
with cores at an unchanged working set, so §18 cannot be the cause.

Together these rule out HBM bandwidth and the spill term. They do **not** rule out a cost for
LX traffic, and that alternative is not tested here. Whole-loop LX traffic on the sweep above
is 2.10 MB at 1, 4 and 8 tiles — as invariant as the element count — so a finite LX bandwidth
predicts the same flat time by the same argument, and LX is per-core, so it would scale with
cores identically. The magnitudes are also plausible: the fitted rate corresponds to about
6 GB/s per core if each element crosses LX once, or 31 GB/s for a five-stage chain touching
it at every stage. Since LX traffic is itself proportional to the element count, the two
stories share a functional form and the current data cannot separate them.

**The deciding experiment, run.** Vary the number of fused stages at a fixed element count
and fixed HBM traffic, at a low core count where the floor binds. More stages mean more LX
traffic and more per-element work; the element count does not move, so the current form
predicts no change at all. Softmax over 2048 × 512 at 8 tiles, sigmoid stages inserted
between the two reductions,, 7 repeats:

| chain | HBM (MB) | 1 core (µs) | 2 cores (µs) |
|---:|---:|---:|---:|
| 5 | 6.3 | 591 | 304 |
| 7 | 7.3 | 829 | 428 |
| 9 | 8.4 | 1056 | 554 |
| 13 | 10.5 | 1522 | 803 |

**Time scales with the chain, and the element-only form is wrong.** From 5 to 13 stages the
time grows **2.57×** at one core and **2.64×** at two, against a chain-length ratio of 2.60
and an element-only prediction of 1.00. Some HBM traffic does leak — 0.52 MB per stage
against a 4.2 MB full round trip, 12 % of one — but that cannot be the driver: the HBM bytes
grow only 1.67× while the time grows 2.57×. Time tracks the chain length to within 1–2 %,
not the bytes.

This does **not** name the mechanism. LX traffic and per-element work through more stages are
both proportional to `elements × stages`, so the sweep separates the *form* from the
element-only one without separating those two from each other.

At 32 cores the time grows 5.57×, faster than the chain, and none of the three forms fits.
The floor does not bind there, so that regime is governed by other terms.

**Model, and what is now known to be wrong with it.** As shipped, a fused reduction cannot
finish faster than its elements allow:

```text
time ≥ elements / (cores × 1.51 elements per nanosecond per core)
```

taken as a floor under the byte-based estimate, where `elements` is the largest operand the
chain touches. One parameter, fitted over 97 repeat-backed runs. It closes the shape above
from −86 % to **+6.5 %**.

The experiment above shows this form is mis-keyed: it has no chain-length term, so it holds a
13-stage chain to the same floor as a 5-stage one and under-predicts the long chain by a
factor of 2.6. Every run it was fitted on is a 5-stage softmax, so the chain length is
silently folded into the constant — `1.51` is really a per-stage rate divided by five. The
correction is not applied here: it would need a re-fit across the coarse categories, and the
floor never binds at 32 cores, so nothing in the target configuration moves either way.

**How well understood is this term?** *The form is empirical; the mechanism is open.* That
the cost is not the HBM byte count is settled — the byte-only model is 5× low before tiling
is varied at all, and the alternative of a per-core **HBM** bandwidth derate was tried and is
worse (five parameters, biased −27 % on its own calibration data, 1.6× worse on unseen
shapes). What sets the rate is not settled: per-core element throughput and a finite LX
bandwidth fit the same data equally well, and the experiment above has not been run. The rate
of 1.51 elements per nanosecond per core is fitted with no derivation from device
specifications. The practical stakes are low — the floor never binds at 32 cores, so it
changes nothing on a full machine (0 of 1871 predictions).

### Part IV data — every coarse-tiling run

Every coarse-tiling measurement in the band the accuracy claim covers: at least 2048 rows
and 2048 columns, on all 32 cores, excluding flash attention (a fused attention kernel, out of scope here). Runs whose recorded byte
counts predate the per-tensor move-count fix of §15 are excluded, because scoring any model
against a byte count known to be wrong measures nothing; that exclusion is applied by a
structural test (a tiled multiply must have a repeating operand) rather than by naming
logs. Predictions are recomputed from the live model, not read from storage.

#### Accuracy, by operation

| op | n | RMS % | mean % | worst % | >20 % |
|---|---:|---:|---:|---:|---:|
| `matmul_row_tiling` | 136 | 7.5 | -1.9 | +19.1 | 0 |
| `softmax_row_tiling` | 64 | 7.2 | -0.3 | +21.7 | 1 |
| `softmax_noexp_row_tiling` | 3 | 5.3 | +5.3 | +5.9 | 0 |
| `matmul_k_tiling` | 1 | 7.7 | -7.7 | -7.7 | 0 |
| `mm_nested_m_k` | 1 | 7.7 | -7.7 | -7.7 | 0 |
| **all** | **205** | **7.4** | **-1.3** | **+21.7** | **1** |

| \|err\| band | count |
|---|---:|
| 0-5 % | 96 |
| 5-10 % | 71 |
| 10-15 % | 28 |
| 15-20 % | 9 |
| 20-&infin; % | 1 |

#### Runs in that band

‼ marks a run beyond ±20 %.  `t` is the number of tiles the loop runs.

| op | M | K | N | tiles | rows per core | measured µs | predicted µs | err % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

A representative 28 of the 205 scored runs; all of them: `python3 tools/cost_model/coarse_tables.py`.

| `matmul_k_tiling` | 4096 | 2048 | 2048 | 1 | 512 | 846.2 | 781.2 | -7.7 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 2 | 128 | 340.8 | 362.4 | +6.3 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 4 | 64 | 440.5 | 427.8 | -2.9 |
| `matmul_row_tiling` | 2048 | 2048 | 2048 | 8 | 32 | 653.4 | 595.5 | -8.9 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 2 | 256 | 752.4 | 758.0 | +0.8 |
| `matmul_row_tiling` | 2048 | 2048 | 4096 | 8 | 64 | 1131.3 | 1136.7 | +0.5 |
| `matmul_row_tiling` | 4096 | 2048 | 512 | 1 | 512 | 206.0 | 241.4 | +17.2 |
| `matmul_row_tiling` | 4096 | 2048 | 1024 | 4 | 128 | 435.3 | 395.7 | -9.1 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 1 | 512 | 844.8 | 781.2 | -7.5 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 2 | 256 | 783.5 | 707.9 | -9.7 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 4 | 128 | 677.7 | 719.5 | +6.2 |
| `matmul_row_tiling` | 4096 | 2048 | 2048 | 8 | 64 | 872.1 | 846.5 | -2.9 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 1 | 1024 | 1577.3 | 1450.6 | -8.0 |
| `matmul_row_tiling` | 4096 | 2048 | 4096 | 8 | 128 | 1450.3 | 1602.5 | +10.5 |
| `matmul_row_tiling` | 4096 | 4096 | 2048 | 1 | 512 | 1477.8 | 1398.5 | -5.4 |
| `matmul_row_tiling` | 4096 | 4096 | 4096 | 2 | 512 | 2621.4 | 2575.5 | -1.8 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 1 | 1024 | 1627.6 | 1582.0 | -2.8 |
| `matmul_row_tiling` | 8192 | 2048 | 2048 | 4 | 256 | 1576.1 | 1403.4 | -11.0 |
| `matmul_row_tiling` | 16384 | 2048 | 2048 | 2 | 1024 | 3265.1 | 2934.6 | -10.1 |
| `softmax_noexp_row_tiling` | 16384 | 2048 | - | 16 | - | 1284.0 | 1347.4 | +4.9 |
| `softmax_row_tiling` | 4096 | 2048 | - | 1 | - | 263.0 | 320.0 | +21.7 ‼ |
| `softmax_row_tiling` | 4096 | 2048 | - | 8 | - | 360.1 | 336.8 | -6.5 |
| `softmax_row_tiling` | 6144 | 4096 | - | 2 | - | 1160.4 | 1191.6 | +2.7 |
| `softmax_row_tiling` | 8192 | 2048 | - | 8 | - | 665.6 | 673.7 | +1.2 |
| `softmax_row_tiling` | 8192 | 2048 | - | 16 | - | 676.0 | 673.7 | -0.3 |
| `softmax_row_tiling` | 16384 | 2048 | - | 4 | - | 1544.4 | 1495.0 | -3.2 |
| `softmax_row_tiling` | 16384 | 2048 | - | 16 | - | 1337.5 | 1347.4 | +0.7 |
| `softmax_row_tiling` | 16384 | 4096 | - | 4 | - | 3132.8 | 3317.6 | +5.9 |

The single run beyond ±20 % is a **data** conflict rather than a model failure: the same
configuration measures 389.8 µs in one log and 263.0 µs in another — 48 % apart, with each
log internally stable to under 1 %. Of the 52 configurations measured in more than one log,
51 agree within 10 %. Two operations, `matmul_k_tiling` and `mm_nested_m_k`, contribute one
scoreable run each; the rest of their measurements carry the stale byte counts described
above and are being re-collected.

### Appendix — reproducibility

- **Offline scoring:** `tools/cost_model/eval_model.py` recomputes accuracy for any model version from
  the stored `(features, measured_time)` dataset — no hardware. `--params k=v` re-scores a
  proposed parameter instantly.
- **Figures:** `tools/cost_model/plot_report.py` regenerates every figure from `sweep_records.json`.
- **Sweeps:** each section's data comes from the profiling sweeps under
  `docs/source/user_guide/examples/` (a master runner chains them and folds the results into
  `sweep_records.json`).
