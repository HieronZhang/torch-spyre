# Spyre Cost Model — Status / Handoff (updated 2026-08-03)

Living status + detail doc — **read this first on resume**. The full model write-up is
[cost_model_report.md](cost_model_report.md) (the MAIN doc — the long-form derivation of every
term, with figures and accuracy). This file holds the details that don't belong there:
implementation state, open work, methodology, tooling, next steps. Keep the two in sync — if a
number changes here, update the report.

Goal: a HIGH-LEVEL **relative** cost model over the after-pre-scheduling LoopLevel IR, to
guide optimization (LX placement, coarse-tiling). Bar: correct ranking + ±15-20%. No local HW —
edit locally, hand commands to a run machine, logs pasted back.

---

## ⭐ CURRENT STATE (2026-08-03) — coarse tiling: root cause found, rate blocked on one sweep

Supersedes every dated section below. **Order of work is fixed by the user: coarse tiling is the
prerequisite for flash-attention, so it is solved first.** A deciding sweep is IN FLIGHT (below);
while it runs, the single-op report sections (Parts I–III) are being brought up to date.

### The defect — PROVEN from IR, not inferred from timing

In `haoyang_logs/ir/coarsemm_matmul_row_tiling_2048x2048x2048_t4.txt` the coarse-tiled matmul's
`inner_fn` is:

```text
i0, i1 = index                                # ranges=[2048, 2048]
tmp0 = ops.load(arg0_1, r0_0 + 2048 * i0)     # A[M,K] -- index CONTAINS i0
tmp1 = ops.load(arg1_1, i1   + 2048 * r0_0)   # B[K,N] -- index has NO i0
```

The op carries `loop_tiled_dims=[[0]]` and `DimHint(dim_names=['M'], loop_var=d0)`, so the tiled
symbol is `i0`. Therefore **A advances with the loop and B is loop-INVARIANT** — every iteration
re-enters the same K×N operand. The extractor charges B **once**, because `loop_factor` is computed
as two PER-OP scalars (`out_factor` / `in_factor` in `dump_cost_model.py`) instead of per arg:

```python
out_factor = 1 if tiles_out_dim else loop_trip
in_factor  = 1 if (tiles_out_dim or is_tiled_red) else loop_trip
```

Consequence: **the loop trip count never enters the memory term.** At M=4096 K=N=2048 cores=32 the
model predicts a byte-identical 684.6 µs at L=4, 8 **and** 16 while measurement goes
677.8 → 874.3 → 1306.8 µs. The error ladder is monotone and is **zero exactly where the bug is
inert** (L=1): mean error −0.9 / −6.4 / −9.5 / −23.4 / −58.8 % at L = 1/2/4/8/16.

### The architecture this implies (user's framing, and it is correct)

A coarse-tiled region is **one loop nest containing N ops**, not N independent ops. Loop-invariance
is a property of the loop, which is why it cannot be decided op-locally. The IR already carries
every piece needed for a two-pass extractor, and **most of the plumbing already exists**:

- `loop_info.loop_group_id` is the region identity. Verified: `softmax_row_tiling` has **5**
  ComputedBuffers (max → sub/exp → sum → div) all sharing `loop_group_id=(0,)`;
  `matmul_row_tiling` has **1**. (Earlier "12 ops" readings are dump STAGES, not ops.)
- `spyre_kernel.py:605-660` already maps host-range dim → iteration-space symbol (the
  `host_to_it` mapping) — the same mapping pass 2 needs.
- `ArgTraffic.loop_factor` is **already per-arg and already applied per-arg**:
  `_fused_hbm_bytes` computes `a.elems * a.loop_factor * o.dtype_bytes`.
- `_fused_hbm_bytes(ops)` already takes the whole bundle and dedups external inputs across it.

So pass 2 is: `loop_factor = 1 if the arg's index contains a tiled symbol else loop_trip`.

### What is settled vs what is NOT

**Settled: the COUNT.** B is re-entered every iteration and is charged once. That is a bug.
Validated offline — `cost_model.py` is pure Python and recorded `feats` carry per-arg
`loop_factor`, so **no hardware is needed to test it** (`notes/test_loop_invariant_reread.py`; the
`alpha=0` control reproduces the known ladder exactly). Charging the re-read takes
`matmul_row_tiling` from RMS 21.0 % → **9.1 %**, mean −13.4 % → −1.8 % (repeat-backed subset,
n=77).

**NOT settled: the RATE.** Three effects all scale as `(L−1)` and are **perfectly aliased** on the
data we hold:

| effect | scales as | |
|---|---|---|
| (a) B re-read | `(L−1)·K·N·dtype / BW` | scales with B |
| (b) fixed per-iteration cost | `(L−1)·c` | independent of B |
| (c) per-tile underfill | rows-per-core | independent of B |

Every one of the recorded `matmul_row_tiling` rows has **K = N = 2048 — one single B size
(8.0 MB) across only two (M,K,N,cores) cells.** B never varies, so any coefficient fitted here
would be fitting the aliasing, not measuring a mechanism. This is the same trap as the γ work.

**Also not a monotone effect.** The ladder is **U-shaped**: at M=4096 tiling is 19 % *faster* at
L=4 than at L=1 before turning over. The model already reproduces the down-slope (its spill/LX
behaviour); only the **up-slope** is missing. The fix is an ADDED term, not a replacement.

### ✅ RE-READ LADDER — RUN AND FOLDED (2026-08-03). The RATE is measured; an ONSET is the new unknown

`haoyang_logs/coarse_reread_20260803_211314.log`, 32/32 runs, reps=5, cv ≤ 0.9 %. Folded as
`coarse_rereadnorm_20260803_211314.log` → **2716 → 2748 records**.

**ANSWER TO THE DISCRIMINATOR: the cost scales with B — it is the re-read, not a fixed
per-iteration cost.** Measuring the marginal cost in the *rising* region,
`d = (T(16) − T(8))/8`, at cores = 32:

| N | B = K·N | d µs/iter | B / 150 GB/s | ratio |
|---:|---:|---:|---:|---:|
| 512 | 2 MB | 17.34 | 13.98 | 1.24 |
| 1024 | 4 MB | 27.63 | 27.96 | **0.99** |
| 2048 | 8 MB | 54.09 | 55.92 | **0.97** |
| 4096 | 16 MB | 100.53 | 111.85 | **0.90** |

`d` spans **5.8×** while B spans 8×; a fixed per-iteration cost would have been flat. So each extra
iteration in the rising region costs **one full HBM pass over B at ≈ peak bandwidth** — the count
fix is right and the rate is ~1.0, *not* the 0.5 the earlier offline fit suggested.

**Why the earlier offline fit said 0.5, and why it is not evidence against this.** That test added
bytes through the model's machinery (`/ eff`, `/ s_lx`, then `− γ·min`), which absorbs part of any
added traffic; and its reference for "a full HBM read" was each run's *whole-kernel* `bw_gbps`
(~65–112 GB/s, which includes compute) instead of the 150 GB/s peak. Both errors pushed the
apparent price of a re-read down.

**THE NEW UNKNOWN — the re-read is not paid at every iteration.** Adding `(L−1)·B/peak` to the
run-time predictions makes the fit *worse* (RMS 19.5 % → 23.7 %, mean −10.6 % → +17.9 %), because
the ladder is U-shaped and the up-slope has an **onset**:

| cores | N | T(1) | T(4) | T(8) | T(16) | onset |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 512 | 206.6 | 206.6 | 275.0 | 413.7 | L=8 |
| 32 | 1024 | 417.2 | 433.5 | 439.5 | 660.5 | L=4 |
| 32 | 2048 | 839.0 | 679.8 | 874.9 | 1307.7 | L=8 |
| 32 | 4096 | 1572.8 | 1513.6 | 1445.1 | 2249.3 | L=16 |
| 16 | 4096 | 2598.0 | 2569.4 | 2550.9 | 2521.5 | never (≤16) |

Below the onset the re-read is evidently served on-chip and costs nothing; above it, full price.
**Larger N turns up LATER**, which is the opposite of a naive "big B spills sooner" story — it is
consistent with tiling's *benefit* (shrinking the A/out tiles into LX) growing with N faster than
the re-read *cost* does, moving the crossover right. 8 cells is too few to fit an onset rule, and
one cell never turns over at all, so **do not model the onset yet.**

**Two defects in my sweep script, both now fixed in-tree, both worth knowing:**

1. It wrote `L=4  SUMMARY ...` (indented, prefixed). `parse_sweep_logs.py` requires `SUMMARY` at
   **column 0**, plus `## section` / `-- label`, so it matched **0 runs** on the first attempt. The
   existing log was salvaged by normalising it; the script now emits the canonical format.
2. It did **not** set `SPYRE_DUMP_COST=1`, so these 32 rows carry **no `feats`** and can only be
   scored against the `pred_us` baked in at run time — they cannot be re-scored against a *changed*
   model. The script now sets it and captures the `IO`/`MODEL`/`MODEL FEATS` blocks. **A re-run is
   needed before these rows can validate a model change**; the rate conclusion above does not
   depend on it, since it comes from measured times only.

### 🛠️ FIX IN PROGRESS (2026-08-03) — model term LANDED (inert); one residual isolated

**Landed in `cost_model.py`: `_loop_reread_bytes` + its charge in `predict_ops`.** Counts the HBM
bytes an operand costs beyond its first pass when it is loop-invariant
(`elems * (loop_factor - 1) * dtype`), and charges them at **peak, AFTER the derates**:

```python
mem_t = max(_fused_floor_ns, p.fill_ns + mem / eff / spill_derate)
mem_t += _loop_reread_bytes(ops) / p.mm_bw_read_gbps
```

Placement is mechanism, not convenience: `eff` models a **short per-tile stream** underfilling the
pipeline, whereas a re-read of an invariant operand is one **large contiguous pass** over the whole
operand — it must not be inflated by `1/eff`. Same reasoning as `_fused_floor_ns`, which is applied
at the same point for the same reason. The matmul branch subtracts the same quantity from `r` so
the bytes are charged exactly once.

**GOLD-SAFE, verified: 0 of 2287 feature-carrying records are touched; every category is
byte-identical** (`bmm` 55.0, `bmm_3d2d` 18.0, `matmul_k` 7.9, OVERALL 224.0 / +5.6). `ruff` clean.

**The gate caught a real defect in my first attempt.** I had assumed `loop_factor` was pinned to 1
everywhere. It is **not**: 91 records already carry `loop_factor > 1`, on the REDUCTION-tiled ops
(`matmul_k_tiling`, `bmm_k_tiling`, `bmm_3d2d_k_tiling`, `ctsum`/`ctamax`/`ctamin`) — and on their
**inputs as well as their outputs**. Those factors mean something different (under K-tiling each
iteration takes a fresh K-slice, so the inputs *advance*; only the accumulator is re-touched) and
are themselves part of the known per-iteration-vs-per-loop inconsistency. Charging them made
`matmul_k` **7.9 → 32.4 %** and `bmm_3d2d` **18.0 → 38.8 %**. The term is now gated to
`is_matmul AND tiles_output_dim AND role == "input"` — exactly the established case.

**Independent confirmation of the mechanism, from the recorded features.** For
`matmul_row_tiling`, `matmul_b_bytes` is the **full `K*N*2` at every L (ratio 1.00)** while
`matmul_a_bytes` is exactly `M*K*2 / L` (1.00 / 0.50 / 0.25 / 0.12 / 0.06 at L = 1/2/4/8/16). The
extractor already records the asymmetry in those two fields; it just does not reflect it in
`loop_factor`, which is what feeds the byte count.

**α = 1.0 is corroborated a second way.** Simulating the extractor fix on the 96 in-scope
`matmul_row_tiling` rows, α = 1.0 gives the **flattest** per-L profile (L4 +16.3, L8 +19.9,
L16 +17.1) — the L-slope is captured. Lower α leaves a slope (α = 0.5: +2.0 / −5.1 / −21.1).

**THE REMAINING RESIDUAL, now isolated: a flat ~17 % over-prediction on every tiled run (L ≥ 4),
absent at L = 1/2.** It is *not* the spill term — driving `mm_spill_cap` from 1.5 to 0 moves only
L=1 and leaves L≥4 unchanged (spill is already ~0 there). It is not `eff` either: `eff` varies
0.95 / 0.72 / 0.45 across L = 4/8/16 while the error stays flat, so `eff` is tracking something
real. The shape that fits is a **saturating benefit the model does not have for matmul** — the
A/output tiles shrinking into LX. The raw data shows exactly that: `R(L) = T(L) − (L−1)·B/peak`
falls and saturates at ~0.56 of its L=1 value (cores 32, N 2048). And `_lx_spill_bw_derate` is
**gated OFF for matmul** (`if any(is_matmul): return 1.0`), so coarse matmul currently carries no
LX term at all. That is the next piece, and it is the same gap the LX-information section below
describes.

**UPDATE (2026-08-03, re-run with `feats`).** The fixed script produced 32 rows with features and
**B genuinely varying** (2/4/8/16 MB), so the model side is testable on unaliased data at last.

**α = 1.0 is now confirmed THREE independent ways:**

1. Raw marginal vs a full HBM pass over B: **0.90–1.24** at cores = 32.
2. It gives the **flattest per-L error profile** in the model (L4/L8/L16 = +12.9/+19.5/+18.3).
3. **Per-cell slope match** — comparing the model's own `(pred(16) − pred(8))/8` with measurement:
   **1.01 / 1.02 / 1.13** at cores = 32 for N = 1024/2048/4096.

The residual scales correctly too: at α = 1.0 the error by operand size is +20.6 / +12.7 / +16.7 /
+17.5 % for B = 2/4/8/16 MB — **flat in B**, so the `(L−1)·B` form is right.

**What is left is a LEVEL error, not a slope error.** At cores = 32 the model sits **19–35 % too
high** on tiled runs (`pred/meas` at L=8: 1.26 / 1.31 / 1.19 / 1.35) while reproducing the
per-iteration slope. Cell-normalised, the tiling-specific factor is **0.866 ± 0.092** and has **no
structural driver** — every candidate correlates weakly on 24 points (log2 N −0.45, cores −0.31,
log2 L −0.17, tile rows/core +0.02, out-tile KB/core −0.14). Fitting a term to that would be
curve-fitting, so nothing more is being fitted until the dense ladder lands.

**One real mis-application found on the way.** `coarse_underfill_eff = min(cap, (rows/13)^0.68)`
with `cap = 0.95` is a **softmax-calibrated** term, and on every one of these matmul rows
`tile_rows_per_core` is **32–512** — far above the knee of 13 — so it *only ever returns its cap*,
charging a flat 5 % memory penalty to tiled matmul while modelling nothing. Its sibling
`_lx_spill_bw_derate` is already gated off for matmul for exactly this reason. Removing the cap
gives RMS 18.2 → **16.6**, mean +12.8 → +11.3. Worth doing, but it is a *fraction* of the level gap,
so it is recorded rather than shipped alone.

**Why α = 0.5 must NOT be shipped even though it scores best** (RMS 10.5, mean 0.0): it averages
two opposing mechanisms with one number. At cores = 32 N = 2048 the L=1→16 *average* marginal is
468.7/15 = 31.2 µs = **0.56** of a full re-read while the L=8→16 *local* marginal is 54.1 µs =
**0.97**. Both are correct measurements of different things; a single α splits the difference and
ties one coefficient to two physical effects — the exact failure the outlier attack exists to avoid.

**NOT yet done: the extractor's per-arg `loop_factor`.** Activating it flips `matmul_row_tiling`
from −14.7 % to +10.8 % mean (RMS 22.6 → 17.5, but >10 % 48 → 62) because the other coarse terms
were silently absorbing the missing traffic. Land it together with the LX-benefit term and re-fit
the pair, on a re-run that carries `feats` — not before.

### ✅ PER-LEVEL `loop_factor` — DERIVED FROM THE NEW IR AND IMPLEMENTED (2026-08-04)

The capture ran (all 7 dumps carry `loop_info`; the `matmul_row_tiling` t=4 control reproduces the
known answer exactly), and it settles the nested case.

**`mm_nested_m_k` is `loop_count=[2, tiles]`** — M always split 2 (outer), K split by `tiles`
(inner) — confirmed at t=2/4/8. So `L = 2 x tiles`. (An earlier note said `L = tiles^2`; that is
WRONG — it matches only at tiles=2.)

```text
loop_info=CoarseTileInfo(loop_group_id=(0, 0), loop_count=[2, 4],
                         loop_tiled_dims          = [[0], []],    # level 0 tiles M   -> i0
                         loop_tiled_reduction_dims= [[],  [0]])   # level 1 tiles K   -> r0_0
tmp0 = ops.load(arg0_1, r0_0 + 2048 * i0)     # A
tmp1 = ops.load(arg1_1, i1   + 2048 * r0_0)   # B
```

**The rule** — an operand repeats at a level whose tiled symbol its index does NOT contain, and is
walked at a level whose symbol it does:

```text
factor(arg) = PRODUCT over levels L of ( loop_count[L] if index has no tiled symbol of L else 1 )
```

**IR-derived vs what the extractor emitted** (4096x2048x2048, t=4, out / A / B):

| op | truth | extractor emitted | |
|---|---|---|---|
| `matmul_k_tiling` | 4 / 1 / 1 | 4 / 1 / 1 | already correct |
| `matmul_row_tiling` | 1 / 1 / **4** | 1 / 1 / **1** | B under-counted |
| `mm_nested_m_k` | **4** / 1 / **2** | **1** / 1 / **1** | out and B under-counted |

Two independent cross-checks. (1) `matmul_k_tiling` is the ONLY coarse op whose factors were
already right — and it is the best-scoring coarse op (7.9 % RMS). (2) `matmul_b_bytes` is `B/4` for
`mm_nested_m_k` (B sliced by K) but the FULL B for `matmul_row_tiling` (not sliced): a different
recorded field encoding the same advance/repeat structure.

`mm_nested_m_k`'s OUTPUT is the case a scalar cannot express — it advances at level 0 (index has
`i0`) and repeats at level 1 (no `r0_0`), so 1*4 = 4. And its B is 2*1 = 2, **not** the total
L = 8: applying the flat single-level rule charges 8, a 4x over-count, which is exactly the
RMS 46.0 % -> 142.8 % blow-up seen earlier.

**IMPLEMENTED** in `dump_cost_model.py`: `_tiled_symbols_per_level` (maps per-level host-range dims
to iteration-space symbols, mirroring `spyre_kernel.py`'s `host_to_it`) and
`_loop_factor_for_index`, wired per arg for both the write and every read. Falls back to the old
per-op scalars when there is no `loop_info`. **Unit-verified 10/10 against the IR-derived factors
above**; `ruff` clean; offline scores unchanged (existing rows score from their recorded `feats`,
so the change is inert until a re-extraction).

**Not yet re-measured.** The new factors only take effect on a fresh run. Expect
`matmul_row_tiling` and `mm_nested_m_k` to move; land together with the underfill-cap change and
re-fit as a pair, per the entanglement noted below.

### 🐛 `SPYRE_DUMP_IR=1` WAS BROKEN — it ABORTED THE COMPILE (fixed 2026-08-04)

Every previous attempt to dump the nested ops produced a ~51-line "stub", and the reason was not
that the dump failed to fire — **it killed the build**:

```text
File "torch_spyre/_inductor/dump_loop_ir.py", line 48, in dump_loop_ir
    from .passes import _format_operations
InductorError: ImportError: cannot import name '_format_operations'
```

Two defects, the second making the first fatal:

1. The formatter moved to **`pass_utils.format_operations`**; `passes._format_operations` no longer
   exists (verified: 0 definitions in `passes.py`).
2. The import sat **outside** the `try/except` whose stated contract is *"a debug dump must never
   break compilation"* — so the `ImportError` escaped and aborted every run with the flag set.

Both fixed: import corrected and moved inside the guard. **This had been silently blocking all IR
capture**, which is why the nested-loop redesign never had evidence. Needs one run to confirm
end-to-end (cannot be executed locally without `_C`; statically verified).

`pass_utils.format_operations` prints exactly what the redesign needs: `layout`, **`allocation`**
(the LX residency dict), `op_it_space_splits`, `dim_hints`, `loop_info`, and `op.data` (the
`inner_fn` with its per-load index expressions).

Capture script: **`docs/source/user_guide/examples/run_nested_ir_capture.sh`** — 7 compiles,
reps=1. It refuses to run if the fix is absent, and **verifies each dump actually contains
`loop_info`**, failing loudly instead of shipping stubs (the check the earlier capture lacked).
It starts with a `matmul_row_tiling` t=4 CONTROL whose answer we already know.

### ⚠️ CORRECTED (2026-08-04): the bad bmm rows are a SPLIT problem, not coarse tiling and not a bmm baseline

Three claims made and then disproved by measurement — recorded so they are not re-derived:

| claim | verdict |
|---|---|
| "`bmm_k_tiling` inherits an untiled-bmm baseline error" | **FALSE.** Plain `bmm_wd` at the identical shape is **−0.2 %** (meas 1843, pred 1838.6). There is no baseline error. |
| "the features are identical, so the model cannot tell them apart" | **FALSE.** They differ (`rows/core` 64 vs 256, `m_split` 16 vs 4). Only the *predictions* coincide. |
| "it is a split-shape gap, not bmm-specific" | **FALSE.** `mmwd` at 16×2 is **10.6 %**; `bmm` at 16×2 is **55.0 %**. It IS bmm-specific. |

What is actually true: applying the coarse-tiling hint makes the planner choose a **lopsided 16×2**
work division instead of the balanced **4×8**, and the hardware runs that **2.06×** slower
(3790 vs 1843 µs, same shape, same per-core area 32768, same cores). The error follows the SPLIT,
not the tiling — a **plain, non-coarse** bmm forced to 16×2 reads **−53.2 %**, matching the coarse
one (−54.1 %). So coarse tiling is the *trigger*, not the defect; the bmm compute rate is, and the
report already flags it as *"calibrated at one split (4×8) and cores = 32"*. bmm accuracy peaks at
rows/core 128–256 (−3.1 / −9.8 %) — the 4×8 neighbourhood — and falls away in both directions
(64 → −42.6 %, 512 → −22.0 %, 1024 → −68.8 %).

**Four candidate fixes killed by testing**, so they are not retried:

1. *Split-aware operand traffic* (`n·|A| + m·|B|`) — refuted on **258 `mmwd` rows across 12
   splits**: residual vs un-charged traffic r = **+0.126**, flat while the traffic ratio spans
   2.67×–25.8×. Operands are shared; `|A|+|B|` is correct.
2. *Relaxing §12's area gate* — `bmm_k_tiling` 55.0 → 52.9 % while `mmwd` regresses 15.1 → 21.4 %.
3. *§12 re-scaled* — even fully ungated it supplies **85.9 µs of the 1947 µs gap (4.4 %)**. It is
   additive and area-proportional; the real effect is multiplicative and area-independent. Wrong
   functional form, not a wrong coefficient.
4. *Array fill / rows-per-core* — `mm` at rows/core = 64 is **16.8 %**, `bmm` at 64 is **56.5 %**.
   Same rows/core, so it is not `pt_eff`.

**What would fix it:** make `_matmul_mac_peak` depend on split geometry as well as layout,
calibrated by a bmm split sweep at fixed shape/batch/cores and a **pinned fast layout** (the layout
constants 650/368/517 were also fitted at 4×8, so varying both leaves them confounded).

**Also worth raising as a COMPILER issue independent of the model:** 4×8 exists, is available, and
is 2.06× faster than the 16×2 the hinted path selects.

### 📊 COARSE TILING — every tested case, scored against the LIVE model (2026-08-03)

In-scope, scoreable, `kernel_us`. Regenerate with the snippet in `notes/report_tables.py`'s
population loader.

| op | tiling | n | RMS % | mean % | worst | >10 % | unscoreable | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `matmul_k_tiling` | reduction (K) | 34 | **7.9** | −5.8 | −13.3 | 6 | 0 | **best coarse op**; near the bar |
| `softmax_noexp_row_tiling` | output (rows) | 3 | **5.3** | +5.3 | +5.9 | 0 | 0 | fine, but n=3 |
| `ctsum` | output (rows) | 2 | **6.0** | +1.9 | +7.6 | 0 | 26 | fine; 26 rows unscoreable |
| `softmax_row_tiling` | output (rows) | 80 | 13.1 | −4.3 | −42.8 | 20 | 13 | shipped elem-throughput floor; residual is the cores 8/16 roofline |
| `bmm_3d2d_k_tiling` | reduction (K) | 8 | 18.0 | −17.1 | −26.0 | 7 | 0 | two-rate step shipped; still under |
| `matmul_row_tiling` | output (M) | 128 | 21.8 | −13.7 | −64.8 | 64 | 42 | **mechanism solved** (invariant-B re-read, α=1.0 confirmed 3×); fix staged, level residual ~13 % open |
| `mm_nested_m_k` | nested (M+K) | 28 | 41.4 | −32.9 | −71.8 | 18 | 0 | same sign as row-tiling but the row-tiling rule makes it WORSE — needs its own IR |
| `bmm_k_tiling` | reduction (K) | 17 | 55.0 | −54.1 | −67.9 | 17 | 0 | **NOT a coarse defect** — its UNTILED rows are already −38…−52 %; it inherits the bmm baseline error |
| `bmm_nested_b_k` | nested (B+K) | 7 | 59.7 | **+26.3** | +94.3 | 7 | 0 | two opposite errors: the same −50 % bmm baseline at L=1, plus a model that grows ~4× too steeply with L |
| `flash_attn` | fused attention | 7 | **3521** | +3386 | +4431 | 7 | 0 | ~35× over-prediction; blocked behind coarse tiling by design |
| `ctamax` / `ctamin` | output (rows) | 0 | — | — | — | — | 4 + 4 | no scoreable rows |
| `softmax_unrolled` | output (rows) | 0 | — | — | — | — | — | entirely out of scope (runs at cores=1 by design, < the cores ≥ 8 rule) |
| **ALL COARSE** | | **314** | **526** | +62.5 | +4431 | **146** | | dominated by flash |

**Reading it.** Excluding flash, coarse tiling is **~25 % RMS over 307 points with 139 beyond
±10 %** — the weakest part of the model, and the reason it is the prerequisite for flash.
Three things stand out:

1. **The two bad bmm rows are NOT coarse-tiling failures — check the UNTILED row first.**
   `bmm_k_tiling` at `tiles=1` is already **−51.5 / −38.8 / −51.6 / −49.3 / −50.6 %**, i.e. ~−50 %
   before any tiling exists, whereas `matmul_k_tiling` at `tiles=1` is **+1.2 / −7.5 / +2.3 / +2.6
   / +1.7 / +2.9 %**. Same tiling mode, and the difference is entirely in the *baseline*. Both bmm
   coarse ops inherit the untiled-bmm error behind `bmm_wd`'s −26.4 % mean; tiling adds ~15 pp on
   top. **Fixing coarse tiling will not fix them.** Diagnose per op by reading its `tiles=1` row
   before attributing anything to the loop.
2. **`bmm_nested_b_k` over-predicts because two opposite errors superimpose.** Measured grows
   **1.9×** from L=1→16 (3794.8 → 7138.3 µs) while the model grows **7.3×** (1838.6 → 13414.5). At
   L=1 it is the same −51.5 % baseline as `bmm_k_tiling` — identical config, identical prediction.
   The steepness cannot be the compute term (`matmul_macs` *shrinks* with L here, per-tile), so it
   is memory-side; `eff` falling with tile height is the leading candidate, unconfirmed. Note
   L = tiles² here (nested B×K), so it moves faster than any other op.
3. Two coarse cohorts pulling in opposite directions is exactly why the γ re-fit was refused
   earlier — they cannot both be arbiters, and now we know at least one of them is not really a
   coarse measurement at all.
4. **42 of 170 `matmul_row_tiling` rows are unscoreable**, along with 26 `ctsum` and 13
   `softmax_row_tiling`. That is a data-hygiene gap independent of any model term.

### 🔎 IS B IN LX? NO — settled two independent ways (2026-08-03)

**From the compiler, not from timing.** A graph input can only become LX-resident by being
**cloned** into LX (`allocator.py`: the source "stays in HBM (not an LX candidate)"). Whether that
clone is made is decided by `_input_residency_reason`, whose FIRST gate is:

```python
if self._read_count(uses) == 0:
    return "no consumer reads it from LX"
```

and `_read_count(uses) = max(0, len(uses) - 1)` — **the first use is never counted**, because it is
the clone-in read the input cannot avoid. A coarse-tiled matmul is **one** ComputedBuffer, so B has
exactly **one** consuming op ⇒ `read_count = 0` ⇒ residency denied. Pinning it would cost a
clone-in transfer and save nothing *as far as the allocator can see* — the allocator counts uses in
the GRAPH, and the L loop iterations are inside a single op, invisible to it.

**Empirically, from every record we hold.** Coarse-matmul input args: **684 HBM / 0 LX**.
`matmul_row_tiling`: **0 LX args in 363**. `mmwd`: 0 LX in 1260. The planner is not broken and was
not off (`LX_PLANNING=1`, `lx=1` in every SUMMARY) — `softmax_row_tiling` gets **1420** LX args. It
simply declines for matmul.

**So "B is read every iteration" is the better-supported reading, not the opposite.** B is in HBM
and nothing pins it. Subtracting exactly one re-read per iteration,
`R(L) = T(L) − (L−1)·B/peak`, leaves a **monotone falling, saturating** curve in **5 of 8** cells —
the signature of a separate benefit (the A/output tiles shrinking into LX), not of a re-read that
is not happening. In the 3 exceptions the marginal *exceeds* one full re-read (1.18–1.24), which is
the opposite of "read less often"; those are the smallest B (2 MB at both core counts, 4 MB at 16
cores), where a short transfer does not reach peak bandwidth.

### 📌 THE COST MODEL CAN GET MUCH BETTER LX INFORMATION — and mostly does not use it

What the model does **today**: `_mem_of_layout` reduces `layout.allocation` to a **boolean**
lx/hbm; `_lx_spill_working_set` **estimates** the per-core working set as "~2 live tiles of
`tile_rows_per_core × cols`"; `_lx_spill_bw_derate` compares that against a **fitted**
`lx_spill_cap_bytes = 512 KB` (the real budget is 1638 KB/core = 2 MB × (1 − 0.2)); and the whole
term is **gated OFF for matmul** (`if any(is_matmul): return 1.0`). So for coarse matmul the model
carries no LX term at all, and everywhere else it is approximating a quantity the planner computed
exactly.

What is available and unused:

| source | what it gives | reachable today? |
|---|---|---|
| `layout.allocation["lx"]` | the LX **address** of a placed buffer — with sizes, the real per-buffer footprint and true occupancy | **yes**, but read only as a boolean |
| `allocator.reject_reasons: dict[str, str]` | per-buffer verdict for **why** a buffer is not in LX: `no consumer reads it from LX`, `partial/offset read`, `core div mismatch: …`, `lx back gap`, restickify barrier | **no** — computed every compile, emitted only at DEBUG (`_log_lx_pinning`), never stamped on the op |
| solver `log_lx_usage` | per-timestep scratchpad occupancy | **no** — DEBUG only |

**The change is small and contained:** stamp `reject_reasons` (and the allocation address/size)
onto the ops the way `loop_info` is already stamped, then read them in `dump_cost_model`. The model
would then *know* that B was denied LX and why, instead of inferring spill from a size heuristic —
which is precisely the input the re-read term needs, and would also let the `s_lx` derate use the
planner's real answer rather than a fitted 512 KB proxy.

### 🔬 ORIGINAL SWEEP DESIGN — `run_coarse_reread_ladder.sh` (launched 2026-08-03)

`docs/source/user_guide/examples/run_coarse_reread_ladder.sh` — 32 runs, reps=5, cores ≥ 8.
Two knobs break the aliasing: **N** varies B=K·N without touching A=M·K; **cores** varies
rows-per-core without touching B. Discriminator = the per-iteration slope
`s = (T(L) − T(1)) / (L − 1)` plotted against N:

- `s` ∝ N → the re-read is real and HBM-priced.
- `s` flat in N → it is a fixed per-iteration cost; do **not** add re-read bytes.
- `s` sub-linear in N → partial residency; `s / (B/BW)` **measures** the HBM fraction of each
  re-read, which is the number that goes in the model.

Env vars and op name verified against `profile_ops.py`; `bash -n` clean. **UNTRACKED — must be
COPIED to the run machine (rsync/scp), not pulled.**

### Do NOT generalise the fix to the nested ops

The same patch makes `mm_nested_m_k` / `bmm_nested_b_k` **worse** (RMS 46.0 % → 142.8 %). Their
loops are nested (`loop_group_id=(0,0)`, `loop_count=[K1,K2]`) and tile **K as well as M**, so B is
sliced along K and is *not* invariant. Each op needs its own IR dump before any change — per-op
verification is doing real work here.

### Next steps, in order

1. Fold the re-read ladder; read `s` vs N; fix the rate **or** conclude it is a fixed per-iteration
   cost. Re-run `notes/test_loop_invariant_reread.py` with real B variation.
2. Implement the two-pass extractor (pass 1 group by `loop_group_id` → tiled symbols; pass 2
   per-arg `loop_factor`). Gold-safety gate + adversarial review as usual.
3. Dump IR for the nested ops, then decide their rule separately.
4. Only then flash-attention (currently ~40× over-predicted).

### Known scoring gaps found while re-scoring (2026-08-03)

- ⚠️ **`sweep_records.json` IS PRIMARY DATA — never rebuild it by re-parsing `haoyang_logs/*`.**
  **10 of the 51 curated logs no longer exist on disk, and 189 records survive only inside the
  JSON.** A blanket re-parse silently drops them and simultaneously re-admits ~31 superseded /
  duplicate logs the curation had excluded (five `new_experiments_20260721_*` at 143 rows each,
  repeated `outlier_*` attempts, …), which is how the file went 2748 → 4237 on 2026-08-03. The
  correct update is always **fold the NAMED new log only**:
  `python3 notes/parse_sweep_logs.py haoyang_logs/<the-new-log>.log` — the parser merges keyed by
  `log_file:lineno`, so this is idempotent and touches nothing else. Back the file up before any
  bulk operation. Current curated state: **2780 records / 51 logs / 2062 in scope / 1780
  scoreable**.
- **`kernel_us` is the canonical measured field** (present on all 2038 in-scope rows and used by
  `eval_model.py`). `kernel_us_min` exists on only 45 % (repeat-backed rows) and runs a median
  0.77 % below it. Any ad-hoc analysis must use `kernel_us` or its `n` will silently be a
  repeat-biased subset.
- **Plain `mm` is unscoreable**: all 21 records are `no-feats` (no `feats`, no reconstructable I/O
  block), so §7–11 — the sections that derive the core matmul memory/compute/overlap terms — have
  **zero** rows scoring against the live model. Those terms are in practice validated on `mmwd`
  (§12, 318 rows). There is also no plain-`mm` IR dump in `haoyang_logs/ir/`. Worth closing.

---

## CURRENT STATE & PLAN (2026-07-24) — superseded by the section above

**The active effort is the "outlier attack": fix every ≥10% mispredicted *normal/valid* input,
mechanistically and noise-controlled** (standing directive, see memory `outlier-attack-six-categories`;
detailed plan `~/.claude/plans/shimmering-percolating-rivest.md`). Ignore tiny matmuls and
1×32/32×1 extremes.

### 🏁 AUTONOMOUS RUN — ENDED (was 2026-07-24, session 2) — cats 3→6 then flash-attn

**This run is over; no heartbeat is active.** Kept for its discipline checklist, which still
applies. Its findings were superseded by sessions 3–5 and by the CURRENT STATE section at the top.

Ran unattended (~4h). A 5-min heartbeat cron re-invoked and read THIS section to continue.
Discipline (MUST): mechanism-based modeling; **adversarially review EVERY claim with a Workflow of
challenger agents before acting** (user mandate, repeated); design isolation sweeps + hand off if
data is thin (don't wait); dump lower-level IR locally if mechanism unclear; goal <10%/point; never
commit/git; regenerate figure + full end-of-section table after any model change; keep report+status
consistent. Analysis scripts: `notes/analyze_matmul_overlap.py` (cat3), `notes/analyze_bmm_layout.py`
(cat4). **Findings below are UNDER REVIEW / not yet verified until their challenge workflow passes.**

### TASK 8 (report figure fixes) — DONE (2026-07-24), review COMPLETED (no longer in flight)

User-added task: fix three report figures/claims. All done on EXISTING data (no HW), each
independently re-verified against `sweep_records.json` before editing; a Task-8 challenge workflow
is reviewing the revisions.

- **§3 add4 (fig3b):** the fused-vs-separate +0.31 gap at add4 is REAL (cv 0%, n≥5, all 7 shapes)
  but the earlier figure had the DIRECTION backwards. Verified: the **fused** chain is a clean line
  (add4 residual +0.07; model predicts fused add4 to **+2.1%**); it is **add4_sep** that dips ~0.2
  BELOW the trend (~2/3 of the gap). Fuser code (`fusion.py::_max_bundle_tensors` = len(SEGMENT_OFFSETS,
  7) −1 = 6; `spyre_fuse_nodes`) shows add4 = 5 non-intermediate tensors = ONE left-associative bundle;
  first split at add5/add6, **never add4** → refutes the reassociation/barrier hypothesis. The effect
  lives in the SEPARATE multi-launch control (host launch-scheduling regime, a HYPOTHESIS). NOT
  modelled. Confirmatory sweep written: `docs/source/user_guide/examples/run_add_chain_ir.sh`
  (add3–6 + _sep, SPYRE_DUMP_IR=1, grep `async_compile.sdsc(` bundle count).
- **§8 fig8 (memory term):** the two literal requests are WRONG for this figure — "drop K≤64" would
  delete the whole write-heavy corner; "add 8×4/4×8/16×2 marked" tests nothing because the memory
  term is **split-agnostic** (`cost_model.py::_fused_hbm_bytes` returns TOTAL bytes, no m/n/k). Did:
  dropped only the DEGENERATE K=64 point (`2048×64×2048`=56.6µs, below K=16/32 despite more bytes),
  kept K=16/32; added prose that split coverage lives in §9/§11/§12. Live errs: write-heavy ≤4%,
  read-heavy +8–18%.
- **§9 fig9 (compute term):** the claim "4×8/8×4/2×16 all ≈385µs" is FALSE. Verified 32-core splits:
  K2048 {4×8=387,8×4=386} collapse, 2×16=404(+4.7%), 16×2=396; K4096 {4×8,8×4,16×2}≈670 collapse,
  2×16=704(+5%), 1×32/32×1≈2×. Rewrote fig9: every split at true (1/cores,t), balanced/thin markers,
  peak line fit on balanced-only, + a 32-core zoom INSET so 2×16 is visible (the "missing green 2×16"
  was crushed by the 0–4000µs range + a mean-merge annotation, not filtered). Prose now agrees with the
  §11–§12 work-div table (2×16 measured ~399µs).

### ✅ SESSION 3 (2026-07-29) — outlier close-out on EXISTING data: 3 shipped, 3 refused

Scored the live model over **normal** inputs (cores=32, non-lopsided split, min(M,N)≥512): 465 pts
>10 %, of which **164 are the chained adds** (§3's documented program-level effect, out of scope by
user decision) → **301 in scope**. Worked only what existing repeat-backed data can settle.

**SHIPPED (each gold-safe, verified by me — max|Δ| = 0.000000 outside its target set):**

1. **Cat 4 — bmm layout penalties are ADDITIVE PER OPERAND** (the mechanism, not just a rate).
   Across all **11 matched quads** (byte-identical, copy-free, every cell reps=7),
   `(A-slow + B-slow)/(both-slow + both-fast)` = **0.983** → the reciprocal rates add:
   `1/peak = 1/650 + [A default]/368 + [B default]/517`. Three params reproduce ALL FOUR combos to a
   few percent where the previous single constant could only reproduce the both-default *sum* —
   which is why the *mixed* layouts had been the worst-predicted points in the model. Per-combo
   mean|err|: def/def 74→**1.5**, def/fast 64→**2.5**, fast/def 58→**2.5**, fast/fast 24→**4.4** %.
   46-shape clean table mean|err| **4.1 %**. Gold-safe over 1612 non-target records. Code:
   `_bmm_layout_pair` (ordered pair; A/B order verified against recorded layout_a/layout_b 54/54),
   `_matmul_mac_peak`, params `bmm_layout_peak_{fast,a_default,b_default}_ns`.

   **Corrections after adversarial review (all verified by me):**
   - The additivity ratio is **not** "1.0 within noise" as first written. The run-to-run floor is
     ~0.225 %; 8 of 11 quads exceed 3σ, 9 of 11 sit **below** 1, and the residual tracks output width
     (`r(log2 N) = −0.77`). There is a real **~1.7 % super-additivity** the form knowingly drops. The
     range `[0.953, 1.015]` also held only under the most favourable rep aggregation; medians/means
     give `[0.9547, 1.0179]`.
   - **The "74 → 1.5 %" comparison is against the pre-cat-4 model with no bmm term, not against what
     was actually replaced.** Every bmm the compiler emits today is **both-default** — all 99 real
     in-gate rows — so df/fd/ff exist only in the synthetic `bmm_layout` experiment. Measured against
     the 160 constant this term replaced, production is **neutral**: mean|err| 24.98 → **25.01 %**
     (0.03 pt, far below noise), and on the dd combo itself 5.51 → **5.49 %**. The constants were
     therefore **re-fit in relative-TIME space with dd weighted** (was: uniform error in inverse-peak
     space, which over-weights the fast combos), moving 635/375/524 → **650/368/517** and the implied
     dd rate to **161.5** (measured 161.3, old constant 160). The term's value is pricing a layout
     *change*, not scoring today's kernels — stated as such in code and report.
   - The additive **form** survives every attack: multiplicative is off 26–57 %; four free per-combo
     rates gain ≤0.9 %; leave-one-shape-out actually **favours** additive (4.52 vs 4.57 %) — so it is
     not over-parameterised. Weakest point: `peak_fast` is anchored on the `ff` combo (worst
     residuals, mean|e| 4.4 %) which **no real op produces**, yet supplies ~25 % of the dd rate.
   - The review's **"unmodelled B trend inside dd"** (backed-out peak 158.9/161.6/165.1 at B=4/8/16)
     **does NOT survive a confound check — I overturned it.** The shape MIX changes with B: B=16 is
     measured only at `1024x2048x1024` and `512x2048x512`, and the latter alone implies ~170, which
     drags the B=16 median up. Shape-matched, the batch effect is **inconsistent in sign and tiny**:
     `1024x2048x1024` 161.3→162.2→162.7 (B=4/8/16, +0.9 % total), `1024x2048x2048` 158.2→160.9,
     `2048x2048x1024` 158.1→158.0 (**flat**), `512x2048x512` 170.3→169.8 (**flat**) — 2 of 4 matched
     pairs flat. Within-cell noise floor is 0.22, so the +0.9 % on the one shape spanning all three
     batch sizes is real but negligible. **What is actually unmodelled is a SHAPE spread**: at fixed B
     the implied rate runs 158.0…170.3 (**7.8 %**, ~8× the shape-matched batch effect), with
     `512x2048x512` the fast outlier. **NEW, and a correction to my own first reading:** that shape
     spread is worth only ~±4 % of prediction error and does **NOT** explain the §13 table's +41 %
     row. At `512x2048x512`, cores=32, dd: the `bmm_layout` runs are **+4.6 %** (B=8) and **+4.3 %**
     (B=16), while the plain `bmm_wd` run is **+40.7 %** (B=4) — *same* per-core tile (128 rows x
     **64 cols = exactly ONE STICK**, the narrowest possible output tile), same cores, same combo,
     ~10x the error. **CONFOUND NOW BROKEN with existing data — the op/harness explanation is
     REFUTED.** Six other (shape,B) cells were measured BOTH ways (`bmm_wd` and `bmm_layout`, dd,
     cores>=8) and the two agree to **+0.2 / +0.1 / -0.5 / +0.1 / -0.1 / +0.0 pp** — they are the same
     measurement. So the +40.7 % is NOT a harness artifact; what remains is a **small-shape effect at
     the smallest batch** (B=4 vs 8/16). Array underfill at a one-stick tile is the natural suspect
     and matches the plain-matmul signature, but it rests on a **SINGLE record** (n=1), so it stays
     flagged, NOT modelled. **Already covered by the widened sweep**: BMMFULL now runs `bmm_layout` at
     B in {2,4,8} on `512x2048x512` with repeats, which confirms or kills it outright. So the queued ladder should vary **shape at
     fixed B**, not B; a per-B term would fit a composition artifact.
   - Bookkeeping: the joint diff had **2** regressions (not 1) and **41** records crossing >10 → ≤10
     (not 35); and the gold-safety statement covers the two shipped terms **separately** — 6 of the
     moved records belong to the `transpose_outer` term, not this one.
2. **Cat 2 — `transpose_outer` M<8 penalty.** M is the output's contiguous stick-run length, so M<8
   ⇒ sub-1 KB writes. Measured (all reps=7, cv<1 %): M=8 +3.7 %, M=4 −16.0 %, M=2 −26.8 % — monotone
   in log2(M). Charged as **13 GB/s per halving below M=8**, applied AFTER the surface clamp (the
   floor is calibrated at M=8). transpose_outer RMS **9.6→7.1 %**, >10 % 13→7, worst M<8
   **29.4→8.9 %**. §6 transport category 7.0→**5.9 %**. M>8 deliberately NOT modelled (weaker,
   R-dependent, confounded with a planner split-shape change).
3. **Scoring-harness bug** — `eval_model.reconstruct_from_io` dropped `logical`/`dims`, so feats-less
   rows silently missed `_transport_kind` and were scored on the WRONG model. Now carried through.

**REFUSED / DROPPED after investigation (each recorded so it is not re-attempted):**

- **γ re-centre REFUSED.** Implied γ≈0.58 on the repeat-backed cores-ladder tempted a change, and
  γ 0.62–0.66 does improve the 104-pt clean cohort (8.71→6.97 % RMS). But it **regresses everything
  else**: matmul_split 14.3→14.7, matmul_k 7.2→8.7, matmul_row 23.7→26.4, OVERALL 43.2→43.5. The
  entanglement trap for the second time — **shipped γ=0.46/peak=1140 stays**.
- **`copy`/`bcastcol` regime boundary DROPPED.** The proposed premise did not reproduce (claimed
  `copy 512×4096` at −10.7 %; actual **+5 %**). The genuinely low-cv broadcast outliers are almost
  all `write` (out of scope — its error sign flips in both R and C, a 2-D surface no power law
  expresses); the remaining cbc outliers sit at **cv 3–13 %**, i.e. noise-dominated.
- **`cat0` / `cat1` / `write`** — 3 cells / 1 sub-512 point / no mechanism respectively. Not fitted.

**OVERLAP FORM re-derived from `max(compute, mem)` (user request) — shipped form SURVIVES.**
Key identity: `compute + mem − γ·min(compute,mem)` **≡** `max(compute,mem) + (1−γ)·min(compute,mem)`,
so the model already IS "max + xxx" with `xxx = 0.54·min`. The real question is the FORM of `xxx`,
which is why re-scaling γ kept failing. Tested (components extracted exactly by solving the live
model at two parameter settings, `write` from `write_bytes`), scored on three nested cohorts:

| form | reps-backed (104) | ALL mm/mmwd (343) | ALL matmul-family (755) |
|---|---:|---:|---:|
| **F0 shipped `max + 0.54·min`** | 8.71 | **14.34** (mean **−0.09**) | **28.96** |
| F1 pure `max` | 17.14 | 26.01 | 37.56 |
| F3 `max(compute,read) + write` | 11.91 | 21.73 | 34.15 |
| F3′ `+ 1.85·write` | 7.27 | 20.39 | 31.66 |
| F4 `max + 0.40·min` (γ=0.60) | 7.22 | 14.94 | 30.12 |

(1) **Perfect double-buffering is REFUTED** — pure `max` is far worse everywhere, so the
non-overlapped component is real. (2) **No constant fill/drain** — the best-fit constant is c = 0.
(3) **"Stores are serial" is REFUTED at its physical coefficient** (1.0 is worse than shipped
everywhere); only an unphysical 1.85× fits the narrow cohort and it collapses globally (worst 97 %).
(4) The narrow-cohort winners lose globally, and F0 is the only essentially **unbiased** form on the
343-point set. **Implication: the remaining matmul residual is NOT in the overlap form** — it is in
the terms feeding it (compute rate, spill knee, coarse tiling), i.e. exactly where the paused sweeps
point. Do not re-litigate the overlap form without new data.

### ✅ SESSION 5 (2026-07-30) — close-out sweep landed; cat-5 FUSED reduction SHIPPED

**Data.** The full close-out sweep ran on HW (`closeout_20260729_074904.log` + its four companion
logs: `reduction_cores`, `gamma_bind`, `coarse_reduction`, `coarse_mm_tile`). Folded with the
script's own documented command — **306 new rows, 286 repeat-backed**, 2141 → **2447** records.
NOTE: re-parsing *all* logs instead balloons the file to 4633; the 2141 baseline is a CURATED subset,
so always fold the named new logs only.

**CAT 5 FUSED REDUCTION — SHIPPED (the pause is resolved), but NOT with my first mechanism.**
The pause reason was "every low-core `softmax_row_tiling` point is single-shot". The new sweep makes
every core count repeat-backed (n=8/8/8/10/8 at cores 1/2/4/8/16, cv 0.05–1.7 %), and the error was a
clean monotone function of cores: **−90.9 / −85.7 / −72.9 / −57.4 / −45.3 / −3.0 %** at cores
1/2/4/8/16/32, with `softmax_unrolled` (cores=1 by design) at **−91.9 %**.

**⚠️ MY FIRST ATTEMPT WAS REFUTED BY REVIEW — recorded so it is not retried.** I shipped a per-core
BANDWIDTH derate (`softmax_bw_cores_g`, five fitted values), reasoning by analogy with the shipped
plain-reduction term. The adversarial panel returned STANDS-WITH-FIXES on the *existence* of a
cores term but refuted the mechanism, and I verified every load-bearing number myself:

- **Arithmetic bug in my own calibration.** I fit `g = median(pred/meas)` over the WHOLE prediction,
  but the code divided only `(r+w)/bw_peak` by `g` — the turnaround term (27–29 % of `mem`) was
  undivided. Post-fix error on **my own calibration cells** was **−17.2 %** (−25.9/−22.6/−19.8/
  −15.3/−12.3 at cores 1/2/4/8/16) where it should have been ~0. Reproduced exactly.
- **The mechanism itself was wrong.** Three measurements separate throughput from bandwidth:
  (1) the deficit scales as **~1/cores** (4780/1873/791/418/211 µs at cores 1/2/4/8/16 on one matched
  shape) — a per-core rate, not an additive cost; (2) measured time is **FLAT across tiles**
  (5376/5470/5375/4578 µs at tiles 1/4/8/16, cores=1) while the model's counted traffic falls **2.6×**
  (67.1M → 25.2M elements) — so a term keyed on counted bytes cannot be right, and the "TILES
  residual" I had flagged as physics is a **model byte-count artifact**; (3) at iso-working-set
  (LX spill ∝ 1/(cores·tiles), so exact diagonals exist) time still halves with cores
  (4578/2288/1154 µs at (1,16)/(2,8)/(4,4)) — which rules out LX spill.

**SHIPPED INSTEAD: a per-core ELEMENT-THROUGHPUT floor**, one parameter.
`T ≥ elems/(cores · 1.51 elem/ns/core)`, applied to the **final** memory time (applying it to raw
`mem` inflates it by 1/(eff·spill_derate) — that cost me one iteration). The element count is the
largest HBM operand, `== rows*cols` on **186/186** measured bundles.

| | original | my BW table (5 params) | **throughput floor (1 param)** |
|---|---:|---:|---:|
| `softmax_unrolled` | 89.5 % RMS, −89.3 % | 45.1 %, −28.5 % | **14.6 %**, +12.1 % |
| `softmax` | 46.8 %, −32.6 % | 27.9 %, −16.5 % | **21.8 %**, −9.3 % |
| OVERALL | 42.5 % | 39.9 % | **39.2 %** |

Independently verified: best rate **1.51** (matches review), mean error **+0.3 %** on the 97
low-core records, and **leave-one-shape-out RMS(log) 24.7 % vs the table's 39.6 %** — 1.6× better
out-of-sample with 1/5 the parameters. **Gold-safe: 1975 unchanged, 93 moved, 0 at cores≥32**; the
floor never binds at cores=32 (0/89 records), so it cannot touch the gold path by construction.

**FLAGGED, deliberately NOT modelled:** the floor leaves a systematic residual at **cores=8/16**
(median −16.5 % / −41.4 %) where the throughput bound hands back to the memory term and that term is
itself too optimistic — the true form is a roofline whose *other* side also saturates, which needs
the bus term re-derived at the same time. Independently, the memory side still tracks **COLS** at
cores=32 (−50.9 / −38.7 / −17.5 / +0.4 % at cols 128/256/512/2048), which no core-count term reaches.

**Two of my documentation claims were also wrong and are corrected:** the gate does **not** catch the
`ctsum`/`ctamax`/`ctamin` rows (18/18 are intercepted earlier by `hbm_pattern='reduce_outer'`), and
the cores=32/cols=2048 error is **+0.4 %**, not the −5.1 % I wrote. Calibration/scoring also used
different statistics (`kernel_us_min` vs `kernel_us`); the reported +12.1 % mean on
`softmax_unrolled` is partly that.

**THE THREE bmm CORNERS the plan had shelved as "too thin" — re-examined with the new data.**
All gated bmm regimes are already good (`bmm_layout` +0.4 % n=71, `bmm_wd` +0.1 % n=16), so every
remaining bmm error is in these three un-gated corners.

1. **cores<8 — STILL NOT ACTIONABLE (verified, not assumed).** The sweep added only 6 low-core bmm
   points, all one shape. `pt_eff` is **1.000** on all of them, so the existing array-underfill term
   is NOT the cause. But the split geometry changes *with* cores (1x1 at c1 -> 1x2 at c2 -> 1x4 or
   2x2 at c4), so the plan's original objection — cores confounded with per-core area and split
   fanout — still stands at **n=2 per core count on one shape**. Implied per-core rate 410/242/185
   at cores 1/2/4 saturating to 161.5 at >=8 (reproduces the plan's earlier 407/241/168). Do not fit.

2. **B=2 — EFFECT CONFIRMED, MECHANISM NOT.** This is now a clean controlled comparison: at
   `1024x2048x1024` the split is byte-identical across B=2/4/8/16 (4x8, rpc=256, cpc=128,
   reduction_cores=1), so only B differs. Per-batch time is **flat for B>=4 and ~half at B=2**:

   | combo | B=2 | B=4 | B=8 | B=16 | B2/B>=4 |
   |---|---:|---:|---:|---:|---:|
   | dd | **232.5** | 460.3 | 457.9 | 456.8 | 0.507 |
   | df | **187.4** | 322.1 | 318.8 | 319.9 | 0.586 |
   | fd | **150.9** | 265.3 | 272.2 | 266.3 | 0.567 |
   | ff | **107.1** | 137.2 | 146.0 | 146.4 | 0.754 |

   The **additive layout structure still holds** at B=2 with every rate scaled ~2x (implied dd 356,
   df 466, fd 626, ff 1186 vs 161.5/235/288/650 at B>=4). n=19 repeat-backed, cv 0.2-1.0 %.

   *Mechanism hypothesis (NOT confirmed).* Per-core weight footprint = `B*K*(N/n_split)*2` bytes vs
   the 1638 KB LX budget: B=2 -> 1024 KB (fits), B=4 -> 2048 KB (spills), so weights would re-fetch
   per batch above the threshold. The model's own `_lx_spill_bw_derate` is **1.0 with working_set
   0.0 for every B**, i.e. it does not see the bmm weight footprint at all. **But the data cannot
   confirm this**: there is exactly ONE shape whose B=2 footprint exceeds the budget
   (`1024x2048x2048` fd, 2048 KB), and its ratio **0.629 sits INSIDE the spread of the shapes that
   fit (0.507-0.754)**. One discriminating point does not separate the hypothesis, and `ff` at 0.754
   already breaks the clean "2x" story among the fitting shapes. **Deciding experiment queued**
   (`BMMLX` section): B=2 across shapes straddling the 1638 KB threshold at fixed split.

3. **3d2d projection — THE PLAN'S BLOCKER IS RESOLVED; this is the most shippable of the three.**
   The plan refused it because "a flat rate is refuted -- us/GMAC spans 2.7x with a K-dependence
   whose **~178 us intercept implies a FIXED cost**, and B and fixed-cost/MACs are ALIASED". With the
   new repeat-backed K x B data that intercept **does not exist**: regressing time on MACs over 13
   repeat-backed records gives `t = -4.7 us + 72.7 us/GMAC` — **no meaningful fixed cost**, so the
   aliasing objection is gone. What remains is a clean B-dependent RATE, tight within each B:

   | B | n | us/GMAC | range | implied MAC/ns/core |
   |---:|---:|---:|---|---:|
   | 4 | 5 | **57.6** | 56.2-62.1 (4 distinct shapes) | ~542 |
   | 8 | 6 | **80.1** | 75.7-80.2 (+ one 113.3 at K=512) | ~396 |

   vs the plain **1140** the model currently uses for 3d2d and 161.5 for a full bmm.

   **SHIPPED: a TWO-RATE step, `bmm_3d2d_mac_peak_lo_ns = 705` (B<=4) / `hi = 470` (B>8)**,
   with a new `_bmm_3d2d_batch` detector (exactly ONE rank-3 operand). Detector fires on 48
   records by leading op — **62 bundles** once every op in the bundle is considered — all
   `bmm_wd_3d2d`/`bmm_3d2d_k_tiling`, and on no other op (0/573 plain 2D, 0/175 full bmm, 0 rank-4).

   | cohort | before | flat 605 (first attempt) | **step 705/470** |
   |---|---:|---:|---:|
   | calibration (repeat-backed plain 3d2d, n=19) | 38.1 % RMS | 16.5 %, −2.2 % | **5.6 %, −0.2 %** |
   | all plain 3d2d (n=43) | — | 24.1 % | **19.8 %** |
   | OVERALL | 39.2 % | 38.9 % | **38.8 %** |

   Gold-safe: **2006 unchanged, 62 moved, 0 without a 3d2d op**; exactly **one** repeat-backed
   record anywhere regressed >2pp (see residual below).

   **⚠️ MY FIRST VERSION WAS A FLAT RATE AND THE REVIEW REFUTED IT — recorded so it is not retried.**
   I shipped flat 605 and justified skipping the B dependence as "confounded". **Three of the four
   supporting facts were wrong**, and I verified every correction myself:

   - *"Only two shapes have >1 B value"* — **FALSE, there are three**, and `1024x2048x1024` is a
     fully repeat-backed FOUR-point ladder (B=2/4/8/16, all reps=7).
   - *"B is confounded with shape family"* — **FALSE**. Within shape, at an identical 4x8 split,
     per-GMAC cost normalised to B=4 replicates the same non-monotone curve three times:
     1024x1024x1024 → 1.136/1.000/1.352/1.384 and 1024x2048x1024 → 1.145/1.000/1.425/1.415.
   - *"The deciding experiment still needs running"* — **it was already in the database**, in the
     HEAD `closeout_20260729_074904.log` section literally titled "3d2d rate": six reps=7 rows,
     B∈{2,4,8} × 2 shapes. **I excluded them by filtering on `M/K/N`, which are unparsed (None) for
     those rows — the shape is only in the label.** At flat 605 they show a monotone sign flip
     **+6.7/+4.9 → +19.0/+15.6 → −17.5/−16.3**; under the step they sit at **−4.9 … +6.1 %**.
   - Only *"not capacity-driven"* survived — and it argues FOR modelling B, since equal working
     sets with different B give 57.6 vs 80.1 us/GMAC.

   Leave-one-SHAPE-out (holding out a whole B ladder) confirms it: step **21.3 %** vs flat 25.1 %
   on my cohort, and the reviewer's tighter cohort gives 9.2 % vs 18.0 %. **I under-fit.**

   **THE STATED MECHANISM WAS ALSO WRONG and is now removed.** "The 2D operand loads once and is
   reused" is already paid for: that operand is `broadcast=True`, `loop_factor=1`, so the byte
   count charges it **once** — charging it again as a rate double-books. Two further contradictions:
   all **43/43** measured rank-3 operands are in the SLOW default order, for which the layout term's
   own additive rate is **235** vs the 605–705 actually sustained (two shipped mechanisms disagreeing
   about one configuration); and amortisation should improve with more batches, whereas the rate
   **steps down** above B=4. The term now ships as an **empirical rate with an open mechanism**.

   **Other review fixes applied:** `bmm_3d2d_k_tiling` **excluded from calibration** (its
   `matmul_macs` is per-tile, up to 16x under-counted, and fitting those rows alone gives 946 —
   1.6x the plain-3d2d rate, itself evidence they measure something else); the record-count
   ambiguity (48 by leading op vs 62 by bundle) spelled out; the **no-cores-gate decision**
   documented (every measured row is already cores>=8, and the fallback below it would be the plain
   1140, which the sibling's low-core data shows is 2.5-7x too fast).

   **RESIDUAL, disclosed:** `512x2048x512` at B=4 (reps=7, cv 0.89 %) is the one repeat-backed
   record this term makes worse, 4.3 → **10.4 %** — it wants ~965, so the small-shape corner is
   priced by neither rate. The flat version left it at +21.9 %.

**CAT 6 (`matmul_row` / `matmul_nested`) — INVESTIGATED; the `matmul_macs` bug is now fully
characterised, is NOT fixable in the cost model, and is NOT the main cause of either residual.**

*Correction to my own earlier reading.* I had classified the macs semantics from `feats[0]`. For
`mm_nested_m_k` and `matmul_k_tiling` **`feats[0]` is a Pointwise, not the matmul** (`is_matmul=False`,
`macs=0`); the matmul sits at index 1 of a 3–4 op bundle. Scanning every op in the bundle gives the
exact rule, checked on **814 records with a decidable ground truth**:

> `matmul_macs x loop_trip == TOTAL` for **every** coarse op **except `matmul_row_tiling`**, whose
> macs is already TOTAL (ratio 1.000 at every tile count).

There is also an inconsistency **inside a single op's own feature set**: `matmul_row_tiling` records
macs as TOTAL while its own `matmul_a_bytes` and `matmul_rows_per_core` are PER-TILE
(a_bytes/A_total = 0.500/0.250/0.125/0.062 at tiles 2/4/8/16).

*Consequence, measured directly by patching macs offline:*

| | current | `x loop_trip` on the per-tile ops | control: `x loop_trip` on ALL three |
|---|---:|---:|---:|
| `matmul_k_tiling` | 7.3 % RMS, −4.7 % | **4.7 %, −0.5 %** | 4.7 % |
| `matmul_row_tiling` | 23.3 %, −15.0 % | 23.3 % (untouched) | **246 %, +183 %** |
| `mm_nested_m_k` | 39.2 %, −31.6 % | 34.0 %, −27.1 % | 34.0 % |

So the under-count is real — fixing it makes `matmul_k_tiling` essentially **unbiased** — and the
control proves `matmul_row_tiling`'s macs is TOTAL beyond doubt (+183 % if treated otherwise).

*Why it CANNOT be fixed in the cost model.* A feature-only discriminator does exist in principle
(compare `matmul_macs` against the op's own `M_dev*N_dev*K` recovered from `a_bytes`; TOTAL iff the
ratio ≈ `loop_trip`), and it reaches **97.7 %** (795/814). But **9 of the 19 misses are
`matmul_row_tiling` rows classified as PER-TILE**, which is precisely the +183 % failure mode, and it
breaks entirely on bmm ops (ratio 2048–8192, because `matmul_rows_per_core` aliases the BATCH there —
the hazard the extractor's own docstring warns about). A 2.3 % error rate with a catastrophic failure
mode is not shippable. **The fix belongs in the extractor**, which is what the queued `MACSIR` IR
capture is for. Confirmed blocked — not deferred out of caution.

*And the headline for cat 6:* **the macs bug does not explain either target.**
`matmul_row_tiling`'s macs is already correct, so its **−15.0 %** (n=112, 68 repeat-backed) is a
wholly separate defect; and correcting `mm_nested_m_k` moves it only 39.2 → 34.0 %, leaving **−27 %**
unexplained. Both are genuine open modelling gaps, not accounting artifacts — and both now have good
repeat-backed data. That is the next target.

**`matmul_row_tiling` (-15.0 %, n=112 / 68 repeat-backed) — MECHANISM FOUND, FIX BLOCKED by the
same feature-semantics defect. Not fitted.**

The residual is **zero at tiles=1** (median −0.2 %, RMS 3.9 %) and grows monotonically with the tile
count (−8.6 / −10.9 / −19.9 / −64.4 % at tiles 2/4/8/16), worst at small per-core row counts
(`rows_per_core` 32 → −45.3 %, 64 → −18.8 %, >=128 → −1…−8 %). On the 2-D (tiles x rows_per_core)
grid **both** variables move it, and it is U-shaped in `rows_per_core` (best near 128) — which is
exactly what the `pt_eff = 1.0` comment predicted would be "too weak to fit". It is still U-shaped
with the new data.

**The controlled ladder identifies the mechanism.** Holding `rows_per_core = 256`, K = N = 2048 and
growing M with the tile count:

| M | tiles | measured µs | µs/tile | model µs | model µs/tile |
|---:|---:|---:|---:|---:|---:|
| 2048 | 1 | 384 | **384** | 393 | 393 |
| 4096 | 2 | 774 | **387** | 713 | 357 |
| 8192 | 4 | 1533 | **383** | 1366 | 342 |
| 16384 | 8 | 3045 | **381** | 2672 | 334 |

**Measured time per tile is exactly flat; the model's falls.** The model counts the weight operand
**once for the whole loop** (R+W goes 12.58M → 71.3M elements = A + output scaling with the weight
held fixed), so it amortises across tiles something the hardware re-reads every tile. Note
`a_bytes`/`b_bytes` stay pinned at 8,388,608 while M grows 8x — they are PER-TILE while `matmul_macs`
is TOTAL, the same internal inconsistency documented above.

**Why it is not shippable.** Charging the weight once per tile fixes the ladder (−7.8 → −1.5,
−11.9 → −5.1, −12.2 → −4.9 %) but **over-corrects globally**: `matmul_row_tiling` 23.3 → **41.7 %**
RMS, mean −15.0 → **+25.9 %**. So the weight is re-read on some configs and stays resident on others
— a residency/spill question. The model HAS a spill term for exactly this, but it cannot be applied
here because its byte inputs are per-tile while the MAC count is total, so the two disagree about
what "one iteration" means. **This is the same root cause as the `matmul_macs` blocker, and it now
blocks a second target — which raises the priority of the queued `MACSIR` extractor fix from
"nice to have" to "the gate on all of cat 6".**

**`mm_nested_m_k` (-27 % after macs correction) — SAME ROOT CAUSE, third manifestation. Cat 6 is
now blocked at ONE defect class, not three separate ones.**

Controlled test at a **fixed** `2048x2048x2048` shape — identical total MACs, only the tiling changes:

| tiles | loop_trip | measured | model | counted HBM elems |
|---:|---:|---:|---:|---:|
| 1 | 1 | 383 us | 393 | 12,582,912 |
| 2 | 4 | **1096** | 761 | **25,165,888** |
| 4 | 8 | **1705** | 1143 | **25,165,888** |

Two facts settle it. First, **tiling the same matmul makes it 2.9x / 4.5x slower** — a large real
effect, not an artifact. Second, **the counted traffic is byte-IDENTICAL at tiles=2 and tiles=4**
while the measurement differs 1.56x, so the model is *structurally blind* to the difference.

The cause is visible in the bundle: nested tiling produces a 4-op bundle (init Pointwise -> mm ->
accumulate Pointwise -> writeback Pointwise) in which the accumulator **round-trips HBM** (op[2] reads
two 2.10M-element operands and writes 2.10M). Every arg carries **`loop_factor = 1`** while
`loop_trip` is 2/4/8 — so that per-iteration round trip is charged **once per loop**. This is the §3
read-after-write effect at coarse-tile granularity, uncounted.

**UNIFIED DIAGNOSIS — cat 6 is one defect, seen three ways.** The extractor does not consistently
express "per iteration" vs "per loop":

1. **`matmul_macs`** — TOTAL for `matmul_row_tiling`, PER-TILE for every other coarse op
   (814 records; +183 % control). Blocks calibration of every coarse rate.
2. **`matmul_a_bytes`/`b_bytes`/`rows_per_core`** — PER-TILE while `macs` is TOTAL *in the same op*.
   Blocks `matmul_row_tiling`'s weight-residency term (charging the weight per tile fixes the
   controlled ladder but over-corrects globally, 23.3 -> 41.7 % RMS).
3. **`loop_factor`** — pinned at 1 while `loop_trip` > 1. Blocks `mm_nested_m_k`, and is the same
   defect behind the fused-softmax "TILES effect" (counted traffic falls 2.6x across tiles while
   measured time is flat).

All three are the SAME question — what does one iteration cost, and what is charged per loop — and
none is fixable in the cost model: a feature-only discriminator for (1) reaches 97.7 % with a +183 %
failure mode. **The queued `MACSIR` IR capture is the single gate on all of cat 6** (121 points
>10 %), which is why its scope has been widened to record `loop_factor` and the per-arg traffic, not
just the MAC count.

**GAMMA re-tested after the bmm fixes — STILL BLOCKED, now by direct test rather than inference.**
Session 4 left γ open, with the note that the bmm and coarse cohorts pulled it the wrong way "for
reasons unrelated to γ". Having since shipped the layout and 3d2d terms (`bmm_3d2d` 42.7 → 30.3 %,
`bmm_split` 45.0 → 40.3 %), the profile was re-run:

| γ | plain mm | bmm | coarse | ALL family |
|---|---:|---:|---:|---:|
| 0.30 | 15.43 | **38.51** | **23.28** | **27.65** |
| 0.46 shipped | 13.95 | 39.01 | 24.68 | 27.80 |
| 0.60 | **13.25** | 39.84 | 26.63 | 28.43 |

argmin: plain **0.60**, bmm **0.30**, coarse **0.30**, all-family **0.30** — **identical to session 4**.
So a material improvement in bmm accuracy moved its γ preference not at all, which means that pull
was never about the terms I fixed. The residual bmm error is concentrated in `bmm_k_tiling` /
`bmm_nested_b_k` — coarse ops, blocked on the same extractor defect as the rest of cat 6. **γ is
therefore downstream of the `MACSIR` capture too**, and `γ = 0.46` stays as the compromise it is.

**Still open after this sweep** (now with data, next in line): `bmm_split` 45.0 % (n=162),
`matmul_nested` 39.3 %, `bmm_nested` 56.9 %, `matmul_row` 23.5 %, `bmm_3d2d` 24.2 % (the K×B sweep
landed), and the γ question — which the coarse/bmm fixes must precede, per session 4.

### 📋 CLOSE-OUT LEDGER (2026-07-29) — every remaining >10 % point has a disposition

Scored the **live** model over all 1782 records: **625 points > 10 %**. The headline number is
misleading on its own, so it is decomposed by *whether the point is even distinguishable from noise*
(standing rule 2) and by whether its category is open:

| bucket | >10 % | repeat-backed | cv < 2 % | status |
|---|---:|---:|---:|---|
| add-chain (§3) | 173 | 0 | 0 | OUT OF SCOPE by user decision |
| coarse matmul/bmm (cat 6) | 121 | 39 | 39 | PAUSED — need HW |
| bmm (cat 4) | 101 | 25 | 25 | PARTLY PAUSED — 3d2d / B=2 / low-core |
| plain matmul (cat 3) | 84 | 15 | 15 | open |
| coarse softmax (cat 5) | 77 | 45 | 45 | PAUSED — fused bw(cores) |
| broadcast/transport (cat 1/2) | 44 | 35 | 24 | open |
| other pointwise/reduction | 24 | 0 | 0 | open |
| flash_attn | 1 | 1 | 1 | OUT OF SCOPE — multi-op program, rank-4 |

**465 of the 625 have no repeat structure at all** and cannot be separated from noise. Filtering to
*open category + repeat-backed + cv < 2 %* leaves **39 truly actionable points**, and every one has a
recorded disposition — **zero unresolved**:

| op | n | worst | disposition |
|---|---:|---:|---|
| `mmwd` | 15 | +35.8 % | **BLOCKED on the queued sweep.** cat-3 split/spill; γ cannot arbitrate until the coarse `matmul_macs` defect (MACSIR) and the MMSPILL column ladders land |
| `transpose_outer` | 7 | −22.2 % | **DO NOT TOUCH (plan).** VERIFIED: all seven are M ≥ 8 (M = 8,8,8,32,64,8,32), i.e. the shipped M<8 term covers its own regime and these are exactly the M>8 side left unmodelled — it is self-contradicting across (R,C) and confounded with a planner split-shape change |
| `write` | 5 | −27.8 % | **DO NOT TOUCH (plan).** Error sign flips in *both* R and C; no power law expresses that surface |
| `copy` | 4 | −62.8 % | **DROPPED by user** ("leave good data in our database and drop it for now") |
| `bcastcol` | 3 | +18.4 % | **DROPPED by user** (same instruction) |
| `cat1` | 2 | −19.8 % | **DO NOT TOUCH (plan).** VERIFIED per point: the −19.8 % cell is R = 256 < 512, outside the normal band; the other is +10.1 % at R = C = 1024 — a single marginal cell, and a refit on one cell is curve-fitting |
| `mulbcast` / `bcast` | 2 | +13.2 % | **DROPPED by user** (same instruction) |
| `cat0` | 1 | +13.8 % | **DO NOT TOUCH (plan).** Only 3 cells at C ≤ 1024 |

**Conclusion: all non-blocked, non-dropped work is complete.** What remains is either explicitly out
of scope, explicitly dropped by the user, explicitly "do not touch" in the approved plan with a stated
reason, or blocked on `run_outlier_closeout.sh`. Note `copy` at −62.8 % (cv 0.20) is the largest
noise-controlled error left in the model and is dropped only by user instruction — worth re-raising if
that instruction is revisited.

### ⚠️ SESSION 4 (2026-07-29) — "xxx may be more complex" (user): FUNCTIONAL FORM held, "SATURATED" REFUTED

> **Read this header, not the old one.** The functional-family half of this session survived: no
> alternative form for the leftover term beats the shipped `(1−γ)·min` out of sample. The
> **"the overlap term is SATURATED" half was WRONG and is withdrawn** — it rested on a harness
> with two bugs (a dead `setattr` on non-existent split params, so `split` was identically 0; and
> `mac_peak=1e12` failing to zero compute on the bmm-gated path, silently dropping 170 bmm
> records), and on an "oracle" that froze every other coefficient even though γ is entangled with
> SPILL. A proper joint 5-fold CV gains **+0.62/+0.53/+0.57** with γ landing at 0.58–0.70 every
> fold, so **γ = 0.46 is not pinned**. It is kept anyway because the per-cohort argmins pull in
> opposite directions (plain 0.60, bmm 0.30, coarse 0.30) — it is a compromise between populations,
> not a settled constant, and the two dissenting cohorts have known defects that make them poor
> arbiters. Do not re-fit γ until the coarse per-arg `loop_factor` defect (see CURRENT STATE) is
> fixed. Everything below is retained as the record of the search.

Follow-up to the above, on the user's note that the leftover may need a richer form than
`(1−γ)·min`. Harness: `notes/explore_overlap_forms.py`. Four independent lines, same answer.

**1. Six *structurally different* families, not just rescalings of `min`** — softened roofline
`(c^s+m^s)^(1/s)`; **shared-port** `max(c,m,(c+m)/k)` (compute and DMA contend so the *sum* is
rate-limited, and an unbalanced kernel pays **nothing**, unlike F0); geometric blend
`max + a·lo^p·hi^(1−p)`; balance-dependent `max + lo·(a+b·ρ)`; shared-port + residual; and
`max + lo·(a+b/cores)`. Fitted on the repeat-backed cohort, scored on ALL mm/mmwd: they span
**14.24–15.02 RMS %** — narrower than the repeat-to-repeat spread of some configs. Fitted
**in-sample** at cores ≥ 8 (an upper bound on each): **15.51–16.01**. Nothing separates.

**2. ORACLE BOUND — the decisive number.** Best `(peak, γ)` over `[900,1400] × [0.10,0.90]` chosen by
hindsight on ALL mm/mmwd: **14.34 → 14.26 = 0.08 RMS points.** At cores ≥ 8 (267 of 343 points) the
oracle gain is **0.15 / 0.42 / 0.70** points for cores 8/16/32. At cores 1/2/4 it gains 5.4/2.1/2.1 —
but the argmins are **mutually contradictory** ((1050,0.75) vs (1250,0.30) vs (1350,0.30) vs
(900,0.70)), so that gain is the constants absorbing a different effect, not overlap physics.

**3. Sequential identification (the method the earlier γ attempts lacked) — still refused.**
γ/peak are entangled, so peak was identified **only** on cores=1/2 (compute share 0.86–0.97 ⇒ γ nearly
inert), then γ on cores ≥ 8 → (1090, 0.64). Clear win on repeat-backed mm/mmwd (RMS 8.71 → **6.92**,
>10 % 15 → 10, per-core medians flattened −6.7 → −3.2 and +9.5 → +3.3) but **loses globally**
(14.34 → 15.02, >10 % 84 → 112), the loss **entirely at cores=32**. The cohort conflict is
**COVERAGE, not noise**: repeat-backed has cores median 8 (31.7 % at cores 1–2), single-shot has cores
median 32 (0.4 %). **Peak alone is also refused** (1140→1090 at shipped γ: rb 8.71 → 10.13).

**4. γ(cores) with peak pre-identified** = 0.20/0.60/0.84/0.60/0.56/0.52 for cores 1→32 —
**non-monotone**, and at cores=1 barely identifiable (RMS spans only 1.6→3.7 over γ ∈ [0.2,0.8]).
This **retires the report's standing promise** that a core-count-dependent γ was the natural
refinement: the sweep has now been run and does not support it. §10 rewritten accordingly.

**Two false leads I chased and killed myself** (both lesson-1 back-out traps, recorded so they are not
re-attempted): (a) *"the leftover fraction rises with balance ρ"* — real on clean data (0.356 → 0.392
→ 0.462, noise floor 0.003) but the 2-D (K × ρ) grid shows it is carried **entirely by K ≤ 1024** and
**absent at K = 2048** (largest cell, n=28); (b) *"there is a small-K compute-rate shortfall"* —
implied compute scale 1.14/1.11 at K=512/1024 vs 1.00 at K ≥ 2048, but that back-out fixed `a = 0.36`
when the shipped value is **0.54**, so it was absorbing the wrong leftover; against the **shipped**
model the K=512 row is fine (−6.0…+7.8 %). Self-refuted.

**WHERE THE ERROR ACTUALLY IS.** Above 8 cores the residual (~14 % RMS) is **diffuse** — no
work-division variable exceeds |r| = 0.34 (kernel size −0.335, per-core area −0.282, per-core columns
−0.258, lopsidedness −0.224). One structure survives a repeat control: **few per-core columns**.
Repeat-backed only, RMS **18.5 %** at N/n ≤ 128 (n=11) vs **8.3 %** at 129–512 (n=36) and **7.8 %** at
513–1024 (n=6). N/n = 64 is exactly ONE 64-element stick — the narrowest possible output tile — so
array/stick underfill is the natural reading. **CAVEAT I had to apply to myself:** the apparent
*other* arm of a U-shape (RMS 25.9 % above 1024 columns on all data) has **ZERO repeat-backed points**
— it is single-shot only and is NOT counted. Existing coverage is also thin at the left arm: N/n = 64
has 19 records but only **2** repeat-backed. Hence two new one-variable ladders were added to
`run_outlier_closeout.sh` MMSPILL (+10 runs, ~6 min): **Ladder A** varies per-core columns 64→1024
with per-core rows pinned at 512; **Ladder B** varies per-core rows 128→2048 with columns pinned at
256. If the effect is columns/stick, B stays flat; if it is per-core area, B bends. Same signature as
`matmul_row_tiling`.

**⚠️ CONCLUSION CORRECTED AFTER ADVERSARIAL REVIEW — the "SATURATED" claim was WRONG.**
The review returned **REFUTED** on the parameter half of the claim, and I verified every load-bearing
number myself. Three findings, all reproduced:

1. **Two real bugs in my own harness** (`notes/explore_overlap_forms.py`, now fixed + self-checking):
   (a) `decompose()` passed `split_fixed_ns`/`split_per_core_ns`/`split_lopsided_ns`, **none of which
   exist on `CostParams`** — it is a non-frozen, non-slots dataclass, so `setattr` silently created
   dead attributes and `split` was **identically 0** for every record, folded into `mem`. Harmless for
   the shipped form but NOT for the challengers (p-norm computes `(c^s+(m+sp)^s)^(1/s)`), so every
   alternative was scored on a distorted input. (b) `mac_peak_per_core_ns=1e12` does **not** zero
   compute when the bmm-layout gate fires — that path returns the layout rates and ignores the plain
   peak — silently dropping **170/170 bmm records**, i.e. my "755-record family" cohort was really 585
   and excluded the entire bmm population. Fixing both: reconstruction goes from **174/755 failing,
   worst 29 %** to **0/755 at 0.0000 %**. A `verify_decomposition()` self-check now runs first and
   aborts the script if it ever regresses.
2. **My "oracle bound" was not an oracle.** It varied only `(peak, γ)` with every other coefficient
   frozen. γ is entangled with the **spill** terms, so re-fitting it alone *structurally cannot*
   reveal a gain. Verified by 5-fold CV × 3 seeds on the 343: γ alone **−0.01/+0.04/−0.09**;
   spill alone with γ frozen **−0.07/+0.09/−0.11**; **JOINT γ+spill +0.62/+0.53/+0.57** out-of-fold,
   with γ landing at **0.58–0.70 in every fold**. Joint ≫ sum of marginals = the absorption signature.
   The γ profile is **not flat** (14.25 at 0.46 → 13.56 at 0.60). So **γ=0.46 is NOT pinned by the
   data** and must not be described as settled or saturated.
3. **But the review's direction (γ→0.66) fails the SAME global test that refuted every earlier
   candidate**, which I checked and the reviewer also conceded. Profile with spill re-optimised per
   cohort — **the cohorts want OPPOSITE directions**:

   | γ | plain mm | bmm | coarse | ALL family |
   |---|---:|---:|---:|---:|
   | 0.30 | 15.65 | **40.81** | **23.54** | **28.57** |
   | 0.46 shipped | 14.25 | 41.88 | 24.73 | 28.93 |
   | 0.60 | **13.56** | 43.12 | 26.51 | 29.67 |
   | 0.70 | **13.56** | 44.15 | 28.07 | 30.43 |

   argmin: plain **0.60**, bmm **0.30**, coarse **0.30**, all-family **0.30**. Monotone opposite ways.

**CORRECTED POSITION (what to write and act on).** γ=0.46 is a **compromise between sub-populations
pulling in opposite directions**, not a saturated optimum and not a settled constant. The *functional
family* conclusion still stands (held-out on the 239 single-shot points, no family beats the shipped
live model: P 16.20, SR 16.42, F0-refit 16.96 vs shipped **16.19**) — but the *parameter* conclusion
does not. **Do NOT re-fit γ yet**: the two dissenting cohorts are exactly the two with known
unmodelled defects (bmm at 42 % from a separate cause; coarse with the confirmed `matmul_macs`
per-tile-vs-total bug), so they are poor arbiters of a term worth a fraction of a point.
**DECIDING EXPERIMENT:** fix the coarse `matmul_macs` semantics + land the bmm B-ladder, then re-run
this joint γ×spill profile. Gold-safety is not the obstacle — γ/spill move only 1 non-matmul record
(a flash_attn bundle, 0.19 %).

**Also corrected:** the row labelled "F0 shipped" in my family table was fit at a=0.42 (**γ=0.58**),
so the shipped model was never actually in that comparison; on the 343 the shipped model scores
**14.34**, better than that row's 14.70.

**VERIFIED, act only with new data:**

- **The cat-4 term never fires on flash-attention**: flash's matmuls are **rank-4**
  (`[1,4,2048,128]`) and the classifier requires rank-3. Extending it would extrapolate rank-3 rates
  into a regime with zero measurements — documented in §13 instead.
- **`matmul_macs` semantics are INCONSISTENT across coarse ops — RULE NOW PINNED EXACTLY**
  (session 4, measured over every coarse record using TOTAL = `B*M*K*N` from the run label):

  | op | `tiles_output_dim` | TOTAL / recorded macs | n | verdict |
  |---|---|---|---:|---|
  | `matmul_row_tiling` | **True** | **1.0** at loop_trip 2/4/8/16 | 60 | **TOTAL** |
  | `mm_nested_m_k` | True | == loop_trip | 20 | per-tile |
  | `bmm_nested_b_k` | True | == loop_trip | 6 | per-tile |
  | `matmul_k_tiling` | False | == loop_trip | 30 | per-tile |
  | `bmm_k_tiling` | False | == loop_trip | 18 | per-tile |
  | `bmm_3d2d_k_tiling` | False | == loop_trip | 14 | per-tile |

  `matmul_row_tiling` is the **only** op whose macs already covers all tiles, and the model
  multiplies nothing by `loop_trip` (`compute += matmul_macs / cores / (mac_peak*pt_eff)`), so every
  per-tile op **under-counts compute by `loop_trip`** — the leading suspect for `mm_nested_m_k`'s
  −31 % signed error. **NOT FIXABLE OFFLINE, verified three ways:** (a) `tiles_output_dim` does NOT
  discriminate (row_tiling=True is TOTAL, nested=True is per-tile); (b) deriving TOTAL from
  `M_dev*N_dev*(a_bytes/(dtype*M_dev))` matches on only **298 of 738** rows, because for bmm
  `matmul_rows_per_core` picks up the BATCH, a hazard the extractor's own docstring calls out;
  (c) the existing IR dumps for these two ops are **1–2 line stubs** (`SPYRE_DUMP_IR` never fired;
  a real dump is ~1032 lines). **DECIDING EXPERIMENT written and queued:**
  `docs/source/user_guide/examples/run_coarse_macs_ir.sh` (6 runs, ~2 min, wired into the close-out
  sweep as section `MACSIR`) captures real IR for both ops at tiles 1/4/8 so the layout size and
  `reduction_ranges` can be compared directly — that says whether to fix the extractor or add a
  feature. This is on the **critical path for γ**: the coarse cohort is one of the two pulling γ the
  wrong way and cannot arbitrate while its multiply count is wrong.
- **(superseded note)** — per-tile for `matmul_k_tiling`
  (macs=total/tiles, loop_trip=tiles) and `mm_nested_m_k` (total/4 at tiles=2), but **already total**
  for `matmul_row_tiling`. A blanket `×loop_trip` would fix the first two and break the third by
  `tiles×`, and **no feature distinguishes them** (`tiles_output_dim=True` for both). This is an
  EXTRACTOR bug that must be fixed at the source — and it independently confirms **cat 6 must stay
  paused**: its compute feature is not trustworthy.

### ✅ SESSION 2 (2026-07-24 pm) RESULTS — clean reps=7 mm_family data folded, cats 3–6 resolved

The clean forced-core sweep (`mm_family_20260724_082545.log`) is folded. Outcomes (all
adversarially reviewed + self-verified; details in the per-cat lines below):

- **Cat 3 — SHELVED the Part-III rework.** Shipped model is already 6.0% RMS (all <12%) on clean data;
  the rework gives 5.6% (within noise) and peak 1046 REGRESSES to 9.0%. The gain was a noisy-data artifact.
- **Cat 4 — ✅ SHIPPED** the default-layout-bmm slow COMPUTE-rate term (`_matmul_mac_peak`→160 MAC/ns,
  gated B≥4 & cores≥8). Gold-safe (max|Δ|=0 on 1514 non-bmm); clean cohort ~420%→~6%.
- **Cat 5 (fused-reduction bw-cores) — DEFER.** Needed g(cores) is shape-divergent + entangled with the
  already-active underfill & LX-spill terms; all low-core softmax_row_tiling data is SINGLE-SHOT. Sweep:
  `run_coarse_reduction_sweep.sh` (also covers softmax_unrolled's clean ~0.57 ns/out-elem const-rate).
- **Cat 6 (coarse matmul) — DEFER.** matmul_row_tiling is U-shaped + anti-monotonic (LX-residency speedup
  vs per-tile underfill, both invisible since io_hbm is tile-constant) on ~7 thin points → unfittable
  without curve-fit. Sweep WRITTEN: `run_coarse_matmul_tile_sweep.sh` (dense tile ladder, fixed-rpc rows).
- The shipped **cat-5 plain-reduction g(cores)** (session 1) re-validated on the new BWCORES `read` c1 (+2%).

### ⚠️ ADVERSARIAL REVIEW (2026-07-24) — DOWNGRADES that SUPERSEDE the cat 3–7 claims below

A 6-agent challenge panel + noise-weighting lead re-derived every load-bearing number from
`sweep_records.json` + compiler source. The DIRECTIONS mostly survive; several SETTLED-sounding
claims below are downgraded to HYPOTHESIS or REJECTED. **Apply these before the Part III re-fit.**
Critical systemic finding: **the `haoyang_logs/ir/ircap_*` files are STUBS** (kernel_us=1.0, 121–126
bytes, `cores=32` placeholder) from the no-HW/mock runs — so EVERY "confirmed in the IR" claim for
cats 5/6 is actually UNVERIFIED until the real overnight IR lands. Also: only **132/602** matmul-family
records carry repeat structure, ALL model_sha c201383, ALL `is_current=False` — the whole cat-3/4/6
quantitative story rests on that one noisy cohort.

- **Cat 3 — 🛑 CLEAN-DATA VERDICT (2026-07-24 pm): SHELVE the rework, keep the shipped model.** The
  clean reps=7 `MMISO_CORE` sweep (sha fe3de66) is now folded. On it the SHIPPED model (peak 1140,
  γ 0.46) scores **RMS 6.0 %, mean +0.1 %, every point < 12 %**; the full Part-III package is 5.6 %
  (within the 0.8 % cv noise) and peak 1046 alone REGRESSES to 9.0 % (mean −6.5 %). The rework's big
  gain was a noisy-data artifact. NO cat-3 model change. (The detail below is retained as the record.)
- **Cat 3 (old noisy-data analysis):** peak **≈1040** (not 1046 — 1046 only if M=4096's 1073 folds in); peak survives STRONGLY
  (13 single-core pts, CV≈0). γ=0.6 is a **HYPOTHESIS, not a proven constant** — γ is UNIDENTIFIABLE
  on all clean data (low-core pts saturated: overlap=read for any γ≳0.2), binds ONLY on ~30 noisy
  16/32-core pts (CV median 0.55, max 3.87) where the RMS-vs-γ valley is flat 0.4–0.7. What the DATA
  proves is "reads hide" (g=0 → 15% vs 3%), the `min(read,γ·compute)` regime-switch makes a *constant*
  defensible — NOT a measured γ-invariance. **Deciding experiment WRITTEN & ready:**
  `run_gamma_bind_sweep.sh` sweeps small **M=N** (not K — read/compute ∝ cores·(1/M+1/N) is
  K-independent) at cores 16/32 so read/compute straddles γ (512→1.73 … 2560→0.35 at 32c), stick-aligned,
  pt_eff=1, reps=25 to beat the CV~0.55. The old MMISO_CORE (M=N=2048 fixed, varying K) is saturated
  even at 32c (0.43<0.6) so it can NOT pin γ; this sweep can. **The correct re-fit harness is now
  written** (`notes/analyze_matmul_overlap.py`, replacing the stale γ(cores) scratch tool): it
  reproduces the panel (read-overlap+2·min is the best cell; 2·min helps under BOTH overlap forms;
  saturation census γ-blind ≤8c, binds 16/32c) AND refines γ — the entanglement best-fit gives
  **γ*≈0.70–0.78** and the unsaturated-subset valley (n=13, cv~0.8%) is steep below 0.6, flat 0.6–0.8,
  **min ~0.7** → the central value is ~**0.70**, not 0.6 (bounded below at ~0.6; upper side flat). Use
  0.70 as the central value pending the reps-heavy sweep. writes-serial: keep as a modeling choice (never hurts) but
  the "**markedly better / on write-heavy shapes**" justification is REJECTED (0.2–0.3pt, one noisy
  shape). Operand-min spill 2·min SURVIVES (~1.3pt), but the "spill hurts under old form" ENTANGLEMENT
  rationale is REJECTED — 2·min helps under BOTH forms; the true entanglement is **peak↔overlap-form**.
- **Cat 4:** the layout DIRECTION is SOLID (11 byte-identical matched quads, def/best **2.17–3.34×**,
  median 3.13, ~100σ; single-operand swap only 1.5–1.7×, both operands → full ~3×). But **~215 µs/GMAC
  is REJECTED as a flat rate** — it holds only at split m4n8 & B≥4 (B=2→108 anomaly); across splits the
  rate spans 215→2118. The rate must be split-keyed, and the "real bmm = 215-rate" leg rests on
  singletons (93/107 bmm_wd reps=None). HYP. **IR-CONFIRMED (2026-07-24)** from the REAL bmm_layout IR
  (haoyang_logs/ir/bmm_layout_B4_1024x2048x1024_*.txt, NOT stubs): all 4 layout combos have **copies=0**
  (no inserted restickify/clone) and **byte-identical io_hbm=41.9MB** → the delta is PURE dataflow, not
  a copy artifact. Times: A012/B012=1847µs (slow default), A012/B102=1293, A102/B012=1062, A102/B102=556
  (fast) → def/best=**3.32×**; swapping ONLY A→1.74×, only B→1.43×, so the full ~3.3× needs BOTH operands
  on [1,0,2]. Direction fully closed; only the split-keyed RATE (for the shipped term) still needs
  repeat-backed bmm_wd data. **✅ SHIPPED (2026-07-24 pm) after the reps=7 mm_family fold + adversarial
  review:** the "215→2118 spread" was a CONFOUND (mixed coarse bmm_k_tiling + low-core into the flat-rate
  claim). On the isolated clean cohort (bmm_wd + both-default bmm_layout, B≥4, cores=32, reps=7, non-thin,
  16 shapes) **us/GMAC is FLAT ~215 across a 16× MAC range** → it is a slow COMPUTE rate (~160 MAC/ns/core),
  NOT a bandwidth effect. Term: `_matmul_mac_peak` returns `bmm_default_mac_peak_per_core_ns=160` when
  `_default_layout_bmm_batch` (both rank-3 operands batch at device pos −2) ≥ `bmm_default_min_batch=4`
  AND cores ≥ `bmm_default_min_cores=8`, else 1140. **Gold-safe VERIFIED (self, not just agent):
  max|Δ|=0.000000 over all 1514 non-bmm records**; clean cohort mean|err| **~420%→~6%** (n=25 reps-backed).
  Gated corners left honest: low-core bmm (peak 407/241/168 @ c1/2/4 → 160 only c≥8), B=2 (~2× faster,
  ~108 us/GMAC), thin single-stick (M/N≤512), and old SINGLE-SHOT bmm_wd points (distrusted per noise
  protocol). §13 + coeff table regenerated from the live model; ruff clean. Follow-ups (NOT curve-fit
  now): B=2 small-batch rate + a bmm-specific pt_eff for thin tiles + coarse bmm_k_tiling → cat-6.
- **Cat 5:** −91% under-count is SOLID; cores=1 is **CORRECT (by design** — harness sets sencores=1),
  NOT an "extractor bug" → the defect was a missing **BW(cores)** term in the COST MODEL. **PARTIALLY
  SHIPPED (2026-07-24):** a mechanistic `g(cores)` reduction-bandwidth derate (`red_bw_cores_g` +
  `_reduction_bw_cores_factor`, cost_model.py) now scales `reduction_read_bw` below 32 cores
  (g={1:.11,2:.22,4:.43,8:.54,16:.54,32:1}; sub-linear/saturating, g(1)=0.11 not 1/32; falsifies the
  proportional law). Fixes the 20 low-core PLAIN reductions (read/amax/sumrow/mean): mean|err|
  **289%→5.7%**, reduction category RMS **32→4.5**, OVERALL 48.0→47.4. **Gold PROVABLY untouched**
  (g(32)=1 exact; cores=32 reductions + softmax_unrolled + all other categories byte-identical,
  max|delta|=0.0000; verified in the real repo, not just the agent sandbox). §5 report + coeff table
  updated. **softmax_unrolled ITSELF is NOT fixed** (len=5 fused → else branch, structurally excluded)
  — its −90% is a SEPARATE, larger io_hbm effect (LX-resident intermediate over-credited by the byte
  count; a bandwidth derate can't fix it) → deferred to **cat-6** (io_hbm re-crediting for coarse
  fused reductions). Forcing g onto the fused branch FAILS (t1 +51% overshoot, sign-flips across tiles)
  — confirming it's a numerator/byte effect, not bandwidth. Caveat: low-core anchors are single-shot;
  the c8/c16 plateau + shape-generality need `run_reduction_cores_sweep.sh` (WRITTEN, reps=7, both
  aspect ratios). "cores=1 confirmed in IR" → the REAL softmax_unrolled IR (haoyang_logs/ir/
  softmax_unrolled_*.txt, NOT the ircap stubs) confirms cores=1 + CoarseTileInfo loop_count=[tiles].
- **Cat 6:** the cited IR dumps are STUBS. io_hbm-constant holds ONLY for matmul_row (mm_nested &
  matmul_k io_hbm RISE → different sub-problem); and matmul_row time is **U-shaped** (falls then rises),
  not monotonic. mm_nested outer loop_count is 2/4/8 **shape-derived** (a known extractor bug to FIX
  first, not "stuck at 2"). c_fill is NOT grounded (the equivalent c_loop·L term was explicitly removed
  as unvalidated). "Same physics as cat 3" → the supported link is an rpc/tile-height **throughput
  derate (underfill_eff)**, NOT an additive per-tile term. Whole additive-c_fill mechanism → HYP.
  **softmax_unrolled sub-problem CHARACTERIZED (2026-07-24, HYP — not shipped, thin data):** a FUSED
  coarse reduction (softmax) runs at a reduction-like bw that (a) scales with cores (same shared-bus
  effect as the shipped plain-reduction g(cores)) and (b) has a TILED fill/drain penalty. Evidence from
  the real IR+data: softmax_unrolled @ cores=1 — UNTILED (tiles=1) bw **≈24.8** (very consistent, 5
  shapes 24.6–24.9); TILED (tiles≥4) bw **≈11** (tiled/untiled ≈0.44, matches softmax_row_tiling c1=11);
  softmax_row_tiling tiled bw scales **11/20/37/60/75/124** at c1/2/4/8/16/32. The model charges bw_peak
  =150 on the fused (len>1) branch → the −90% miss. NOT shipped because: (i) all the softmax cores-sweep
  points are SINGLETONS (reps=None); (ii) a fused-reduction bw term TOUCHES cores=32 softmax too (the
  "softmax" category, 34.6%), so it is NOT gold-safe like the plain-reduction fix — a category-wide
  change on singletons is exactly the curve-fit-on-thin-data hazard. Deciding experiment WRITTEN:
  `run_coarse_reduction_sweep.sh` (softmax_row_tiling cores×tiles + softmax_unrolled tiles, reps=7 + IR).
  Ship a `(cores-scale × tiled-derate)` fused-reduction bw term only after it confirms under reps AND
  leaves cores=32 within tolerance. **OPPOSITE-SIGN clue (2026-07-24):** the coarse-tiling path
  OVER-predicts softmax_row_tiling by **+21%** (e.g. 16384×4096 t8 c32: meas 2862.9, pred 3454.8 — model
  bw ~116 vs measured 141) while it UNDER-predicts the coarse MATMULS (matmul_row −15%, mm_nested −33%).
  Same coarse path, opposite sign for reductions vs matmuls → a single coarse-tiling fix must reconcile
  BOTH; fitting softmax_row_tiling alone (making it faster) would half-fit and likely worsen matmul_row.
  Confirms cat-6 needs the joint treatment on the REAL coarse-matmul IR, not a per-op patch.
- **Cat 7:** "product 256 > 32 caused the failures" is **REFUTED** — over-subscription is silently
  skipped (not an error); the DOMINANT failure is a ~600s **COMPILE TIMEOUT** (15/18), which the
  diagnosis omitted. Divisibility is ONE verified mode (2/18). run_flash_resweep.sh's log is a MOCK
  (kernel_us=1.0) — it proves nothing compiled; and the guard enforces only product≤32, NOT
  divisibility (satisfied only by the hard-coded ht=4). **FIXED (2026-07-24):** run_flash_resweep.sh
  now carries the corrected diagnosis, adds a per-tile divisibility guard (necessary-condition),
  logs TIMEOUT distinctly from FAILED (rc=124/137), and raises FLASH_TIMEOUT default to 900s.
  **⚠️ CORRECTION (2026-07-29): that "fix" addressed the MINORITY cause.** The user's recollection —
  the flash timeouts were **LX scratchpad exhaustion** — is confirmed by arithmetic. The fused flash
  region holds per-tile `scores`/`exp_scores`, each `[B_t,H_t,Lq_t,Lk_t]` fp16, against only
  ~512 KB/core (`lx_spill_cap_bytes`). Auditing `run_flash_resweep.sh`'s OWN matrix: **10 of 12
  section-A configs and both section-B configs overflow LX even spread over all 32 cores** — e.g.
  the old default `Lq=Lk=4096, ht=8,qt=4,kt=1` needs **2048 KB/core (4× over)**; `ht=1` needs
  16384 KB/core. So the resweep would still have hung. Over-subscription feeds this: a work_div with
  product>32 is silently SKIPPED, so the tile is never divided across cores → per-core set stays huge.
  **Only more coarse tiles shrink the tile; work_div only redistributes it.** Runnable frontier (live
  ≥ 2·tile·2 B, /32 cores): Lq=Lk=4096 needs ht=32,qt=16,kt=8 (16 KB/core) or ht=8,qt=8,kt=2 (512);
  Lq=Lk=2048 → ht=8,qt=8,kt=2 (128 KB); Lq=Lk=1024 → ht≥2 ok. NEW TOOLS (supersede the resweep for
  the "which configs run" question): `flash_probe.py` (pure-arithmetic validator — catches BOTH the
  divisibility error and the LX overflow with no device — plus a single-config compile+time runner)
  and `run_flash_probe.sh` (pre-validates the matrix, then runs each survivor in its OWN timeout'd
  process, since an LX-exhausted run can hang even in teardown). Also: the 45-byte "failed" flash
  captures were MOCK runs (`kernel_us=1.0`), never real failures.
  **IR-CONFIRMED (2026-07-24)** from the 35 REAL flash IR files (haoyang_logs/ir/flash_*.txt): **10 of
  the product-256 `H4-Lq8-Lk8` (>32-core) configs COMPILED** (have op_it_space_splits) → over-subscription
  is silently absorbed, NOT a compile error (diagnosis correction validated); only **2** carry the
  divisibility InductorError (the minority mode); the rest have no IR + no error = the compile TIMEOUTs.
  So the corrected diagnosis (timeout-dominant, product>32 harmless, divisibility minority) is confirmed.

- **Cat 3 (matmul compute/HBM OVERLAP) — RE-SCOPED BY USER (2026-07-24):** the real ask is NOT the
  cores drift and NOT finding another γ-form. **A scalar γ is naive and wrong; the §10 figure shows
  many outliers.** The overlap of HBM-I/O and compute depends on the actual WORKFLOW / ACCESS PATTERN
  (tile structure, how M/N/K and the split m/n/k map to systolic-array passes + operand loads), which
  is currently not understood. **DO NOT brute-force a γ(cores)/γ(shape) form.** Instead: (1) dump the
  LOWER-LEVEL IR (Opexec / fused-kernel / scheduled-kernel) for matmuls of varied shape+split LOCALLY
  (no HW — see the user's directive + `notes/compiler_pipeline_deep_dive.md`) and read HOW compute and
  loads are scheduled/interleaved (double-buffering? pipeline depth? per-tile fill/drain?); (2) read
  the COMPILER's own work-division cost model (`work_division.py`, `_COHORT_LIMIT`) — does it already
  estimate overlap? what drives it?; (3) only then model the overlap mechanistically, WITH evidence;
  if a γ-like form is used, it must have supporting evidence, not just a fit. If data/IR is
  insufficient, design isolation experiments + write the sweep. This SUPERSEDES the "no clean win"
  conclusion below (which only tested γ-forms, the wrong approach).

  **★ MECHANISM FOUND (2026-07-24, measured-data-validated) — the cat-3 answer:**
  The overlap IS **compute-bounded double-buffering**: DeepTools streams the next operand tile from
  LPDDR5→LX while the PT array runs the current tile, so HBM hides UNDER compute but only up to a
  fixed fraction γ of the compute duration. The correct FORM is **`T = compute + mem − min(mem,
  γ·compute)`** (NOT `γ·min(compute,mem)`). Two measured facts make it work:
  (1) **The true sustained peak is ~1046 MAC/ns/core, not 1140** (tight: median 1041, stdev 16, n=37
  across 3 sweep sections; datasheet 1536). The shipped 1140 over-states compute ~9%, and the constant
  γ=0.46 was silently correcting that error AND modeling overlap — **"double duty"**, which is exactly
  why a constant γ looked wrong / access-pattern-dependent (the user's intuition was RIGHT about the
  symptom; the cause is the peak error + the wrong form, not a γ that varies).
  (2) With peak=1046 and the `min(mem, γ·compute)` form, the EFFECTIVE overlap correctly varies with
  the compute/mem balance (access-pattern-dependent) while the UNDERLYING γ is a clean constant ≈0.55
  (the double-buffer window fraction, fit on clean balanced data). Validated on measured balanced
  matmul: **RMS 9.2→8.3%, and the low cores are FIXED (c1 err 7→1, c2 3→1, c4 5→2%)** — the corrected
  peak removes the confound. Spill stays IN the overlappable mem (data REFUTES the "spill is serial"
  guess). Remaining errors are high-core = the §11 SPILL over-charge (separate).
  **IMPLEMENTATION = a coordinated Part III rework (NOT a one-line change):** peak→1046, γ→~0.55, form→
  double-buffering ENTANGLES with §11 (spill) and §12 (split), which were globally fit at γ=0.46+peak
  1140 — at γ=0.55 they must be re-fit (else matmul_split 14.6→15.3, matmul_row 24→28 regress), and the
  bmm/coarse categories lose the accidental γ-masking so they show their TRUE errors (which is HONEST —
  they need the cat-4 layout rate + cat-6 coarse terms). So the overlap fix is the FOUNDATION for cats
  4+6, done together as a Part III re-derivation. Report §10 stays γ=0.46 until the package ships (keep
  code↔report in sync).

  **★★ NOISE-AWARE VERDICT (2026-07-24, adversarial challenge DONE + self-verified):** the overlap
  is **~CONSTANT (γ≈0.55–0.61)** on clean data — the "γ varies with access pattern" appearance is
  MEASUREMENT NOISE, not signal. Verified: within-config γ_eff swing = **0.43 mean** across genuine
  repeats (some configs ±2.6 when a measurement is contended) vs the claimed aspect effect of only
  0.15; and corr(tile_aspect, γ_eff) **collapses +0.91→+0.05** when restricted to repeat-backed
  (n≥2, MIN-meas) configs — the trend lived entirely in noisy single un-repeated points (several with
  CV up to 41%). So NO access-pattern driver beats a constant γ by >~0.7pt held-out (bar was 2pt).
  **The §10 misses are the SPILL term** — it is symmetric `min(1.5, 0.45·log₂(area/area0))·(|A|+|B|)`,
  which OVER-charges tall-operand thin-N shapes and UNDER-charges wide-N thin-K shapes (the
  bidirectional residual) — plus the low-core rate. **Shippable cat-3 model (clean non-batched
  matmul): peak 1046 + double-buffering `min(mem,γ·compute)` + constant γ≈0.61 + an OPERAND-AWARE
  (asymmetric) spill re-fit → RMS 10.3→7.68%.** Still entangled with §12 (split) + bmm (cat 4) for the
  full category. The forced-core sweep (with the 7-rep noise protocol) gives clean γ_eff to CONFIRM
  constancy definitively — that is why it is the decisive experiment. (This process is a model case of
  the review working: sub-challengers "found" a variable-γ trend; the noise-weighting lead reviewer +
  my own check showed it was noise. Do not over-trust a trend built on un-repeated points.)

  **★★★ VALIDATED cat-3 MODEL FORM (2026-07-24) — the implementation spec:**
  `T = compute + read + write + turn − min(read, γ·compute)`, where `compute = MACs/cores/(mac_peak·
  pt_eff)` with **mac_peak = 1046** (was 1140); `read = (operand_bytes + spill)/mm_bw_read` (reads
  double-buffer under compute); `write = output_bytes/mm_bw_write` (output stores are post-compute →
  SERIAL, NOT hidden); `turn = rw_turnaround·min(r,w)`; **γ ≈ 0.61** (double-buffer window fraction —
  overlap hides READS only, up to γ·compute); **spill = 2·min(|A|,|B|)·f(area)** (OPERAND-AWARE: re-read
  bounded by the SMALLER operand; the old symmetric `(|A|+|B|)·f` over-charged tall-operand/thin-N and
  under-charged wide-N — the bidirectional §10 residual). Measured on clean non-batched matmul with
  all pieces together: **RMS 9.2 → 6.2%** (c32 outliers 12→9%). Each piece only works WITH the others
  (2·min spill HURTS under the old γ·min form but HELPS under read-overlap) — hence a JOINT re-fit.
  Non-matmul unaffected (compute=0 → min(read,0)=0). **Blast radius:** shipping this regresses bmm
  (needs cat-4 layout rate) and coarse (needs cat-6) because old γ=0.46+peak1140 masked their true
  errors — so it ships as ONE coordinated Part III change WITH cats 4+6, coefficients FINALIZED on the
  clean forced-core sweep. Code: separate read/write at predict_ops line 972 + peak + γ + spill formula.

  **DEEP INVESTIGATION RESULT (2026-07-24, measured-data-first):**
  (i) The compute/HBM overlap is realized by the PROPRIETARY DeepTools backend (`dxp_standalone`)
  from a static SuperDSC JSON — it is NOT in any dumpable torch-spyre IR (`docs/source/compiler/
  backend.md`). So the overlap CANNOT be read from IR; it must be modeled from MEASURED DATA using
  the front-end-controlled access-pattern drivers (per-core tile M/m×N/n, K, arithmetic intensity).
  (ii) MEASURED `γ_eff=(compute+mem−meas)/min(compute,mem)`, clean cores≥4: **0.58±0.17 overall, but
  0.61±0.086 for NON-SPILLING tiles (area≤65536) vs 0.58±0.18 for spilling.** i.e. the overlap looks
  ~CONSTANT (~0.6) once the §11 spill-term error is removed — the apparent "γ varies with access
  pattern" is largely the spill term leaking into γ_eff. No per-core-tile access-pattern variable
  correlates strongly with γ_eff (best: aspect +0.30; AI −0.22). **Working hypothesis (UNDER
  ADVERSARIAL REVIEW, cuts against the initial prior): the overlap is ~constant (~0.6); the §10
  outliers are the §11 SPILL term + the low-core mac_peak, NOT a non-constant overlap.** If the review
  confirms, the fix is a JOINT re-fit of γ(~0.6)+spill (entangled; §11). The DECISIVE experiment is
  the forced-core sweep `docs/source/user_guide/examples/run_matmul_overlap_iso_sweep.sh` (WRITTEN,
  dry-run OK, 54 runs: cores {1,2,4,8,16,32} balanced across K/M·N + split-shape + batched) — 1 core =
  NO split isolates the overlap from spill/split. Prior γ-form analysis (kept for reference, NEGATIVE):

- **Cat 3 (matmul γ) — PRIOR γ-FORM ANALYSIS (superseded, kept for reference):** ✅ **ANALYZED — no safe model change (reviewed + blast-tested).** The
  challenge workflow + my own blast-radius test settled it. Findings (VERIFIED): (a) no γ(cores) law
  helps (free b≈0); (b) the fill/drain `fd=c·min` is algebraically identical to a constant γ — my
  "fd fails" claim was refuted, but (c) my "clean win = raise γ to 0.58" was ALSO wrong: γ=0.58 helps
  ONLY the balanced-pure-mm slice (clean RMS 9.2→7.7% after dropping the contaminated shape
  4096×2048×4096, whose c4=8045→c8=8255µs is physically impossible) and **REGRESSES** matmul_split/
  _row/_k/bmm (blast test: matmul_split mean −0.3→−5.0). So γ=0.46 is an entangled compromise — DO NOT
  change it. (d) The low-core drift is <10% (not an outlier): cores=1 implied mac_peak≈1046 vs model
  1140 (~8% high, n=11, tight) — a real but sub-threshold isolated derate, deferred. (e) The genuine
  ≥10% outliers are cores=32 SPILL over-charge on tall-operand shapes (8192×2048×1024 +35%→+3% w/o
  spill; 4 of 6 c32 outliers). **Real cat-3 issue = the §11 spill term over-charges when |A| or |B|
  is large** (re-reads the full operand). I ATTEMPTED the fix — 5 candidate spill forms (2·min(|A|,|B|)·f,
  0.5·(|A|+|B|)·f, per-core-tile, 2·√(|A||B|)·f geometric, current). **None cleanly beats current**:
  2·min lowers RMS but raises the >10% count 10→15/43; geometric is ~1pt better RMS at the same 10/43.
  The residual is BIDIRECTIONAL — over-charge on imbalanced M≫N, UNDER-charge on thin-K (a flagged
  regime) — so no single reweighting fixes both; not worth a §11 rework + regression risk for ~1pt on a
  lower-priority term. **Cat-3 = genuine hard residual, no clean win (tried, not skipped).** The big
  matmul errors are elsewhere: **cat 4 (bmm) and cat 6 (coarse: matmul_row −15%, _nested −33%).**
  Report §10 unchanged (γ=0.46 justified). Analysis in `notes/analyze_matmul_overlap.py`.
- **Cat 4 (bmm layout) — HIGH VALUE, findings VERIFIED, model in progress (paused for cat-3 redo):**
  challenge workflow DONE. CONFIRMED (adversarially + re-derived): default `[0,1,2]` tile order is
  ~2-3× SLOWER than `[1,0,2]` at matched shape/MACs/split (bytes/MACs identical, no inserted copy —
  pure dataflow); the **compiler does NOT auto-pick the fast layout (verified in source)** → real bmm
  genuinely uses the slow default → the −68% bmm residual IS the layout penalty (correct to model).
  REFINED by review: it is NOT a `×B` multiplier — the SLOW layout runs at a nearly **constant ~215
  µs/GMAC** on the matched set (B=2 is an anomaly at 109; compute-bound at a fixed slow rate). DETECTOR
  IS CLEAN (verified): `is_matmul` + 2 rank-3 batched inputs + B≥2 + input-A batch at device pos −2
  fires ONLY on full bmm (bmm_wd/k_tiling/nested/layout-default), NOT on plain `mmwd` (326 ops), NOT
  bmm_3d2d, NOT B=1. Model direction: a layout-keyed slow COMPUTE RATE (mac_peak·~0.15) for full-bmm-
  default. On the matched split-4x8 set a compute-derate d≈0.17 → RMS ~25% (B=2 anomaly + still >10%);
  on the varied-split `bmm_wd` set a constant derate scatters (RMS 47%) → there is a SPLIT-shape effect
  on top (like §12). NEXT for cat 4: separate the layout rate from the split effect; likely needs the
  deeper overlap understanding from cat-3 first. `notes/analyze_bmm_layout.py`.
- **Cat 5 (softmax_unrolled):** −93%. Root cause (⚠️ tentative): runs at `sencores=1` BY DESIGN
  (unrolled single-core, `config.unroll_loops=True`); the model uses full HBM BW (150) regardless of
  cores → predicts ~20µs vs ~288µs. This is the **BW(cores)** effect, IN SCOPE here (genuinely 1 core),
  possibly + unaccounted `exp` compute. Needs the BW(cores) calibration + likely a re-run capturing
  `softmax_unrolled` IR (none captured yet). Review pending.
- **Cat 6 (coarse mm/bmm) — CHARACTERIZED (2026-07-24):** splits cleanly. `matmul_k_tiling` −4%
  (working control). `bmm_k_tiling` −66% and `bmm_3d2d_k_tiling` −19% are DOMINATED by the cat-4 bmm
  LAYOUT effect (they're bmm) — cat 4 fixes them, not a coarse term. The genuinely coarse-specific
  misses are **`matmul_row_tiling` −15%** and **`mm_nested_m_k` −33%**. SHARPENED MECHANISM (verified):
  matmul_row_tiling's `io_hbm_bytes` is **CONSTANT across tiles** (25 MB at tiles=1..16) but measured
  time RISES (335→999µs) — so it is NOT a byte re-read and pt_eff is the wrong lever (tested:
  underfill_eff(rpc=32)≈1, barely moved it 24.2→21.6). It is a **PER-TILE PIPELINE FILL/DRAIN loop
  overhead** (~47µs/tile at 2048², larger at 4096²) that scales with `loop_trip` — the model has NO
  per-tile loop cost so it under-predicts as tiles grow. **This is the SAME physics as cat-3's overlap
  fill/drain, just per-coarse-tile** (each tile restarts the PT pipeline → an un-hidden fill/drain ×
  loop_trip) — the report once had a `c_loop·L` term, dropped as "no op exercises it", but coarse
  matmul DOES. mm_nested additionally has `loop_trip` STUCK at 2 regardless of tiles (an extractor
  under-count → dump_cost_model fix). So cat-6 = a per-tile fill/drain term `+ loop_trip·c_fill(tile)`,
  UNIFIED with the cat-3 double-buffering; needs the coarse-tile-count sweep to calibrate `c_fill`.
  Part of the coordinated Part III rework.
- **Cat 7 (flash re-sweep):** ✅ **DIAGNOSED + FIXED.** The overnight flash configs were invalid two
  ways: work_div product 4·8·8=**256 ≫ 32 cores**, and splits that don't divide the tiled buffer dims
  (H/ht=4, buf5 intermediate as small as 2). Fixed re-run `run_flash_resweep.sh` (guarded product≤32,
  small valid splits, IR capture). Data-only (flash not modeled yet).

### 🔑 KEY CROSS-CUTTING FINDING + READY-TO-RUN SWEEPS

**The matmul-family coefficients (cats 3, 4, 6) CANNOT be fit on the overnight data — it lacks the
noise protocol.** Verified twice: cat-3 γ_eff swings ±0.43 across genuine repeats (the whole
constant-vs-variable-γ debate was noise), and cat-6 c_fill residuals swing 641/645/**1260** at one
config. Contended single measurements dominate. So the mechanisms are all FOUND + reviewed, but the
final fit needs clean, repeated data. **Four calibration sweeps are WRITTEN + dry-run-verified**
(all use the BENCH_REPS=7 noise protocol), ready for the user to run:
**★ ONE overnight launcher: `run_overnight_v2.sh`** — runs all three stages in sequence (single
preflight, each stage independent): (1) `run_matmul_family_sweep.sh` [~175 runs, cats 3/4/5/6 timing:
MMISO_CORE forced 1/2/4/8/16/32-core matmul, MMISO_SPLIT, MMISO_BATCH, **BMMLAY** (matched
[0,1,2]²-vs-[1,0,2]² bmm at reps=7 — the clean cat-4 layout-rate source), CTFILL coarse per-tile
fill/drain, BWCORES]; (2) `run_flash_resweep.sh` [cat 7]; (3) IRCAP [25 lower-level IR dumps via
SPYRE_DUMP_IR=1 → CoarseTileInfo/loop_count/op_it_space_splits/device_layout for cats 6/5/4/3 —
adversarially reviewed: configs all run, IR contains the needed structure]. `run_transport_iso_sweep.sh`
(cat 2) already ran + folded. ⚠️ **All these scripts are UNTRACKED (working tree only — user manages
git); COPY them to the run machine (rsync/scp), do NOT rely on git pull, or a stale
`run_matmul_family_sweep.sh` silently drops BMMLAY.** After it folds via `parse_sweep_logs.py`: the
coordinated Part III rework (cats 3+4+6) fits on clean data. Report-ready prose pre-drafted in
`notes/part_iii_rewrite_draft.md`; the validated cat-3 model form + spec are above.

### Data + how to score (do this on resume)

- **Overnight sweep** `haoyang_logs/outlier_20260723_072217.log` (450 usable points) + targeted
  **broadcast small-ROWS sweep** `haoyang_logs/bcast_smallr_20260724_015428.log`, both folded into
  `notes/sweep_records.json`. Per-run IR dumps in `haoyang_logs/ir/`.
- **Noise protocol** now in `profile_ops.py`: `BENCH_REPS` back-to-back profiled measurements →
  `kernel_us_min/median/std/cv` in the SUMMARY.
- ⚠️ **SUPERSEDED (2026-08-03):** this section used to say "`eval_model.py --all` is NOT the
  accuracy number — filter to matched conditions". That is no longer true. The standing scope
  decisions are now enforced inside `eval_model.in_scope()` (cores ≥ 8; fused reductions ≥ 1024
  columns; two corrupt-feature SHAs dropped), so **`eval_model.py --all` IS the accuracy number**
  and every quoted figure refers to the same population (2038 in scope, 1756 scoreable). Per-section report tables come
  from `notes/report_tables.py`. Score on `kernel_us`, **not** `kernel_us_min` — the latter exists
  on only ~45 % of rows, so using it silently reduces any population to a repeat-backed subset.
- Sweep scripts: `docs/source/user_guide/examples/run_outlier_sweep.sh` (the superset, 9 sections,
  budget-guarded via `MAX_SECONDS`), `run_broadcast_smallr_sweep.sh`.

### The six categories — status + key finding

> ⚠️ **The accuracy numbers in this list are 2026-07-24 vintage and are NOT current** — they
> predate the scope decisions, so they were computed over a different population. For live figures
> run `python3 notes/report_tables.py`. Today (1756 scoreable records): broadcast/write **5.7 %**,
> transport **6.1 %**, reduction **7.2 %**, single pointwise **3.6 %**, matmul split **15.1 %**,
> batched matmul **27.3 %** (but **5.9 %** on the fast tile order — the layout we actually target).
> One item below is now outright wrong: transport's `M ≠ 8` is **no longer** unmodelled — the
> `M < 8` penalty (`tx_touter_m_*`) shipped afterwards and holds to within ~6 %; only `M > 8`
> (specifically M=32, mean −12.7 %) is still deliberately unmodelled.

1. **broadcast / write** — ✅ **DONE** (report §4, models below). RMS 12.0→6.3%.
2. **transport** — ✅ **DONE (2026-07-24).** report §6 now 5.5% RMS over 100 shapes (was 8.5%/42;
   all reported outliers fixed). Model: cat0/transpose_outer/cat1 share ONE form
   `clamp(a − b·log₂(sp) − d·log₂R, floor, peak)`, sp=C/64 (the per-row stick-block count); the 32
   cores split sp, effBW falls with the strided per-row gather (more planes) + stride (rows).
   Params `tx_cat0_*`/`tx_touter_*`/`tx_cat1_*`; cat1/transpose_outer routed by `_transport_kind`
   (structural, no re-dump). transpose stays flat 116. **Fixed a PRE-EXISTING bug**: cat0/cat1
   were mis-classified as `write`-like (`_is_outer_broadcast` fired on a single broadcast input) →
   spurious outer-product term; now requires ≥2 size-1 broadcast operands. **Discovery**:
   transpose_outer effBW PEAKS at middle-dim M≈8 (the R×C grid accidentally only sampled M=8, its
   best case); modeled at M=8, M≠8 is a flagged residual (user chose "M=8 only for now"). Details
   in the archived reasoning below (was "IN PROGRESS"):

   Ops understood from device `dims`:
   `transpose` [R,C]→[C,R] = within-stick reshuffle → **flat ~116 GB/s, already ±2% at 32 cores
   (DONE, the −15% is only cores=1, out of scope)**; `transpose_outer` [R,**M**,C]→[M,R,C]
   (M=middle dim, hardcoded 8 → moves 8× the R×C bytes) = whole-stick outer scatter, unmodeled
   (−13…−25%); `cat0` append-rows = strided stick gather (sp=C/64 strided reads/row, stride=R
   sticks), current `stick_scatter` log-fit miscalibrated (over-predicts large C +26%, under mid
   −15%); `cat1` append-stick-dim = rides default, systematic −8…−12% at large C/R. MECHANISM
   (well-supported): effBW falls with **sp = C/64** (the stick-plane count → more strided per-row
   gathers) and mildly with R. NOT yet resolved (needs data): whether the outer-swap count M
   matters (block-transpose vs pure gather), the sp saturation form, and a real R U-shape (both
   small AND large R slow at fixed sp). Isolation sweep WRITTEN + ready:
   `docs/source/user_guide/examples/run_transport_iso_sweep.sh` (46 runs: TOMID M-sweep = the
   crux, SPSAT larger sp, RSHAPE R-extremes, XIR IR capture) + harness knob `TO_MID` in
   `profile_ops.py`. Model AFTER data returns; scratch scorer `notes/analyze_transport.py`.
3. **matmul γ(cores)** — pending. FINDING: error drifts **−6.3% (1 core) → +7.6% (32 cores)** — a
   single scalar γ=0.46 mis-tracks core count. New forced 1/2/4-core data exists (`mmwd`).
4. **bmm layout** — pending, **HIGH VALUE.** FINDING: at matched shape/MACs/split, `[1,0,2]` vs
   default `[0,1,2]` device tile order is **−54…−70%** — a pure dataflow effect (bytes identical,
   no inserted copy). This is the long-unexplained bmm residual. Model a layout-keyed rate. Op
   `bmm_layout` + `WD_LAYOUT_A/B` in the harness.
5. **softmax_unrolled** — pending. −91%: its IR has **no `CoarseTileInfo`**, runs `cores=1`,
   `loop_trip=1` → the extractor under-counts. IR captured in `haoyang_logs/ir/`. It's an
   extractor bug in `dump_cost_model.py`, not a rate.
6. **coarse mm/bmm** — pending, after pure mm/bmm are solid (LX-resident intermediates make the
   overlap harder).

### ⭐⭐ The original directive (VERBATIM, categories 2–6) — the load-bearing spec

The user's exact framing for the remaining categories. **Overarching rule (repeated at the end):
in many of these cases, do NOT use direct mathematical fitting — do mechanism-based modeling.**
Keep this block intact; it is the acceptance bar for each category, not just a to-do list.

> **(2) transport:** There are still some cases:
>
> ```
> cat1              512×8192     243.2    215.9     −11.2
> cat0              8192×8192    8219.3   10011.6   +21.8
> transpose_outer   1024×4096    1477.1   1280.0    −13.3
> transpose_outer   512×8192     1635.7   1280.0    −21.7
> transpose_outer   4096×4096    6010.8   5120.0    −14.8
> transpose_outer   8192×8192    25324.2  20479.8   −19.1
> ```
>
> It seems that we didn't build a good enough model for transpose_outer. What I'm thinking: as we
> wrote, "each output row is reassembled from the 64-element stick blocks of the inputs — a shuffle
> at block granularity — so a wider row has more blocks to permute into place, and that per-block
> shuffle, not the byte count, sets the cost." So we should actually try to model how this shuffle
> effect affects the hbm I/O cost, not just follow the data and build a mathematical model. To
> achieve this, we may need much more data with more coverage. And cat0 and transpose_outer both
> should be somehow affected by this effect and should be properly modeled.
>
> **(3) Matmul.** We can ignore those tiny matrices and very extreme work-division split cases
> (1×32 or 32×1). However, all the other cases we should have a more accurate prediction. According
> to the new figure in figure 10, a lot of normal cases still cannot be accurately predicted with
> our current model. So we have to think more carefully about our model. Some points we have to
> think of: a. why/how we model the overlap of compute and memory I/O. Currently we simply use a
> gamma to estimate how well it can overlap. But this is very naive and I believe can hardly be
> true; especially, I don't think in different cases (e.g., different tensor shapes, or different
> work division splits, batched or not) the gamma are the same value because they could have much
> different access patterns. So we should probably rethink about this first. One thing we should
> definitely do is to design a large set of sweep experiments to isolate the effect of work
> division split by forcing the work to be done with **only one core** and maybe 2, 4 cores (try
> all three of them, especially with only one core). The contents in current section 11 and 12 are
> more convincible, but the data still looks a bit messy. And a big problem in those sections is
> still, as I said, we should probably make better hypothesis about how each of these effects
> affects/hurts the hbm I/O pattern and hence increase hbm or compute cost, we should probably
> really model this instead of doing mathematical regressions. However, I would say section 11 and
> 12 is not as important as broadcast/transport modeling. If it's really hard to model exactly
> these at this time, pure-math modeling based on profile is weakly acceptable.
>
> **(4) batched matmuls.** Currently, nearly all the bmm points look really bad. Currently, we
> still have this problem unresolved: "The residual is over and above the B× accounting — on both
> sides. At equal MACs, the full bmm and the projection differ only in weight traffic" and
> unexplained. A potential explanation is that the device layout of the tensors for bmm is
> currently critically affecting the performance. A bad way of device layout may severely affect
> the performance. We should design comprehensive sweep experiments to address this question and
> verify the hypothesis. To write those corresponding benchmarks, we should use SpyreTensorLayout
> to control the memory layout of the tensors on device. The theory is that [1,0,2] order may
> perform better than the default [0,1,2] order of BMMs. Some constraints: the last dimension is
> the stick dimension. For a mm or bmm to be legal it must be the K dimension of the first argument
> and the N dimension of the second. The compiler will automatically insert a copy of the tensor to
> make this true if given different format. This will confuse the modeling (so avoid it). The
> `to(..., device_layout)` cannot be the very first `to` in your program. The lazy initialization
> of torch spyre is a bit fragile. Be sure to do a normal `to` first and all will be fine. Example:
>
> ```python
> import torch
> from torch_spyre._C import SpyreTensorLayout
>
> DEVICE = torch.device("spyre")
>
> x = torch.rand(3, 128, 256, dtype=torch.float16)
> y = torch.rand(3, 256, 1024, dtype=torch.float16)
>
> @torch.compile
> def bmm(a, b):
>     return a.bmm(b)
>
> # Default layout result
> x_dev = x.to(DEVICE)
> y_dev = y.to(DEVICE)
> result_1 = bmm(x_dev, y_dev).cpu()
>
> # Same as the default memory layout, but specified explicitly
> x_stl = SpyreTensorLayout(x.size(), x.stride(), torch.float16, [0, 1, 2])
> y_stl = SpyreTensorLayout(y.size(), y.stride(), torch.float16, [0, 1, 2])
> x_dev = x.to(DEVICE, device_layout=x_stl)
> y_dev = y.to(DEVICE, device_layout=y_stl)
> result_2 = bmm(x_dev, y_dev).cpu()
>
> # Alternate memory layout: both tensors with dim 1 tiled with 2
> x_stl = SpyreTensorLayout(x.size(), x.stride(), torch.float16, [1, 0, 2])
> y_stl = SpyreTensorLayout(y.size(), y.stride(), torch.float16, [1, 0, 2])
> x_dev = x.to(DEVICE, device_layout=x_stl)
> y_dev = y.to(DEVICE, device_layout=y_stl)
> result_3 = bmm(x_dev, y_dev).cpu()
>
> # You can mix memory layouts as long as the stick dimension is unchanged.
> x_stl = SpyreTensorLayout(x.size(), x.stride(), torch.float16, [1, 0, 2])
> y_stl = SpyreTensorLayout(y.size(), y.stride(), torch.float16, [0, 1, 2])
> x_dev = x.to(DEVICE, device_layout=x_stl)
> y_dev = y.to(DEVICE, device_layout=y_stl)
> result_4 = bmm(x_dev, y_dev).cpu()
>
> torch.testing.assert_close(result_1, result_2, rtol=0.001, atol=0.00001)
> torch.testing.assert_close(result_2, result_3, rtol=0.001, atol=0.00001)
> torch.testing.assert_close(result_3, result_4, rtol=0.001, atol=0.00001)
> ```
>
> **(5) coarse tiling with softmax family (row_tiling, unrolled).** The unrolled part is especially
> bad. Examples:
>
> ```
> softmax_unrolled   1024×512    1  1   288   23   -92
> softmax_unrolled   1024×512    4  1   294   21   -93
> softmax_unrolled   1024×512    8  1   301   21   -93
> softmax_unrolled   1024×512   16  1   339   21   -94
> softmax_unrolled   2048×512    1  1   676  154   -77
> softmax_unrolled   2048×512    8  1   584   42   -93
> softmax_unrolled   2048×512   16  1   596   42   -93
> softmax_unrolled   2048×512   32  1   675   42   -94
> ```
>
> For unrolled, I think we probably have to first make sure that what it is actually doing and we
> are understanding it correctly, so maybe we need much more sweep run examples for it while
> printing out the loop level IR and what inputs our model actually uses and see if there are bugs.
> Besides, we just discussed the multi-op read-write dependency issues. Also examine it for the
> coarse tiling. (Although I think likely it doesn't make much difference here because most of the
> intermediate tensor is located in LX in coarse tiling. However, we also have modeled the cases
> where the LX space is not enough and still some part need to be placed in hbm then it may also
> worth a verification and observation.)
>
> **(6) coarse tiling with matmul or bmm.** Currently we got really bad prediction accuracies. What
> I think is that we should first make sure we fully understand modeling for pure matmul and bmm
> first and then look deeper into coarse tiling. But we should still think carefully about this
> here. As we discussed, gamma may not be able to model how hbm I/O and compute overlap with each
> other. Now since we used coarse tiling, most intermediate tensors are supposed to be located in
> LX, so it makes this overlapping pattern and modeling more complex. To get a clearer picture, we
> may have to design extra sweep experiments that can potentially give us clearer picture about this
> modeling.

**Note (user, emphasized):** in some of these cases you should NOT use direct mathematical fitting
to do the modeling — do more mechanism-based modeling. Pure-math regression is only a
weakly-acceptable fallback for the §11/§12-class (cat 3 non-crux) effects when a mechanism is
genuinely out of reach right now.

### Cat 1 (broadcast/write) — IMPLEMENTED (report §4.1–§4.5)

- **broadcast** — all four ops share ONE surface *shape* (the small-ROWS collapse is a general
  kernel effect, NOT a `b[1,C]`-operand effect — `copy`, a scalar, shows it too); the row-broadcast
  operand only adds a small rate lift. TWO families, each: well-filled surface (ROWS≥1024,
  `a−b·log₂C−c·log₂R`) + short-tensor quadratic (COLS≤4k) / V-valley (COLS≥8k, minimum at
  **ROWS=COLS/64** = the stick-plane count). Params `bcast_*` (row-broadcast) and `cbc_*`
  (scalar/column) in `cost_model.py`. RMS: `bcast`/`mulbcast` **3.4/3.5%**, `copy`/`bcastcol`
  **7.2/7.5%**.
- **write** (outer-product) — refit + capped empirical term
  `min(2.0e-9·ROWS^1.75·COLS^2.6, 2.4·out_bytes)`. **9.6%** (was 18.9%). Still a black-box; worst
  residual `2048×8192` −30% (the power-law is the best simple form — surface/hybrid scored worse).
- Figures `fig4b_write_spill` (all ROWS + model), `fig4c_broadcast_smallr` (the V-valley),
  regenerated by `plot_report.py`.

### Workflow discipline (memories — pull these before report/model work)

`modeling-report-section-workflow` (run-then-claim; verify per-config not RMS; model must depend
on X if data does; mechanism must fit the controls; get-data-don't-guess), `report-writing-style`
(prose + regenerate tables/figures from the live model, never hand-type numbers; revision order),
`claim-discipline-perf-modeling`, `conservative-claims-adversarial-check`, `do-not-commit` (the
user manages ALL git — never commit/push/stage).

### Immediate next step — ⚠️ SUPERSEDED, see CURRENT STATE at the top

> **Do not act on this section.** It is kept as the record of what was planned on 2026-07-24. The
> γ(cores) work it proposes has since been **done and retired**: the sweep was run, and γ(cores)
> came out **non-monotone** (0.20/0.60/0.84/0.60/0.56/0.52 for cores 1→32) and barely identifiable
> at low core counts. The report's old promise that "an `L`-dependent γ is the natural refinement"
> was withdrawn on the strength of it. The live next step is the coarse-tiling per-arg
> `loop_factor` defect.

Cats 1 (broadcast/write) and 2 (transport) are DONE. Start **(3) matmul γ(cores)** — read the
verbatim directive for it above. The single scalar γ=0.46 for compute/HBM overlap can't hold
across shapes/splits/batched-ness; error drifts −6.3% (1 core) → +7.6% (32 cores). New forced
1/2/4-core data exists (`mmwd`, MMCORE sweep section). Model γ(cores) mechanistically (how the
overlap changes with the split), not a bare fit. (Per-category loop: get/confirm clean data →
hypothesize the mechanism → fit its form → verify per-config no-regression → implement → re-score
→ regenerate figures/tables → write the section.)

---

## Golden measurement

AIU profiler: `torch.profiler` PrivateUse1, "Self SPYRE" per-kernel device time (harness
`profile_ops.py`). New image leaves the kernel name BLANK → classify **by exclusion** (device
time not Memset/Memcpy). Do NOT use `SPYRE_PROFILE`/`SPYRE_PROFILE_SYNC` (host wall-clocks).

## Model + params (implemented in torch_spyre/_inductor/cost_model.py)

Form: `T = compute + HBM/eff − γ·min(compute, HBM/eff)` (see presentation §1 for term table).

| param | value | status |
|---|---|---|
| `bw_peak_gbps` / `rw_turnaround_ns_per_byte` | 150 / 0.00574 | ✅ pointwise/reduction ±5% |
| `mm_bw_read_gbps` / `mm_bw_write_gbps` | 150 / 150 | ✅ single-rate (see DONE note §Matmul) |
| `mac_peak_per_core_ns` | **1140** (was 1536) | ✅ compute-dominant low-core fit |
| `overlap_gamma` | **0.46** (NEW) | ✅ jointly fit w/ peak, RMS 1.7% |
| `mm_spill_t0 / slope / cap` | **448 / 1.10 / 1.70** (NEW) | ✅ decouple+reread sweeps |
| `bw_restickify / stick_scatter / reduce_outer` | **116 / shape-dep / 113** | cat0 = 252−4·log2R−12.3·log2C (shape sweep, R²0.93) |
| reduction read rate | **min(150, 114+61·e^(−ROWS/3700))** | ✅ reduction-rows sweep (op-independent, 2.6% RMS) |
| broadcast rate | **SUPERSEDED 2026-07-24** → two-family `BW_eff(R,C)` surface, params `bcast_*`/`cbc_*`. See CURRENT STATE + report §4. RMS 3.4–7.5% |
| `write_reread_coef/r_exp/c_exp` | **2.0e-9 / 1.75 / 2.60 + cap 2.4·out_bytes** (2026-07-24) | ⚠️ EMPIRICAL outer-product term; write 18.9→9.6% (black-box) |
| `pointwise_arity_derate` | **REMOVED** (add3/add4 are chained ops, not a single-op term — see report §3) | — |
| matmul `pt_eff` (`r_full=64`, `exp 0.35`) | matmul only | ✅ unchanged |
| `coarse_underfill` (`r_full 13`, `exp 0.68`, `cap 0.95`) | coarse/softmax only (2026-07-09) | ✅ HW: softmax non-spill RMS 7.2% (spill regime deferred) |
| `psum_per_elem_ns` | **0.14, GATED off matmul** (2026-07-08) | ✅ bug fixed (was +489% on forced `WD_K>1`) |

All matmul + per-op changes are **implemented in cost_model.py + dump_cost_model.py**.
Coarse-tiling terms are NOT reworked yet (open).

### ⚠️ Version hygiene — the recorded `pred_us` is NOT current (2026-07-08)

An adversarial review + direct checks found the dataset **mixes model generations**: `pred_us`
is baked into each log at run time, `sweep_records.csv` spans 14 log-families (June→July), and
26 configs have divergent `pred_us` for identical shapes. The old `db_sweep.log` (the claimed
"validated re-run") **predates the extractor false-positive fix** — `sumall` still shows +37/+40%
and `transpose_outer` +66/+49% in those records — and now also predates the psum gate. **Its raw
file is gone** (records survive only in the CSV). So the earlier "HW-validated ±5% / matmul 8%"
claims are **not backed by current-model records**. Only the measured `kernel_us` is version-
independent and trustworthy.

Parser stamps provenance: `model_sha` (from each sweep's `git:` header), `log_date`, and
`is_current` (= `model_sha == --current-sha`); `--drop-ops chain` retires `chain`.
**RESOLVED 2026-07-09:** a fresh `run_db_sweep.sh` on the current code (sha `078922c`) landed →
**185 `is_current` rows** are now the single clean current-model dataset (filter on `is_current`).
The extractor false-positive fixes are **now confirmed on HW in-record**: `sumall` +2.8/+5.6%
(was +37/+40%), `transpose_outer` −4.6/−14.9% (was +66/+49%), `sumcol` +7.4/−0.7%.

## ⚠️ HISTORICAL (pre-2026-07-23, may be stale) — trust the CURRENT STATE section at the top

Everything from here down is the July 7–10 handoff, **before** the overnight outlier-attack sweep
and the §4 rework. The accuracy claims ("HW-validated ±5%", per-op RMS) predate the noise-controlled
sweep and the current models — verify against `sweep_records.json` before trusting any number below.
Kept for the implementation detail on matmul / coarse-tiling / reductions (still the basis for the
pending categories 3–6). The methodology + tooling sections further down are still valid.

## Findings by category (all HW-validated unless noted — SEE HISTORICAL WARNING ABOVE)

**Pointwise / reduction / broadcast / transport** — ±5% on anchors, but the adversarial review
(2026-07-08) found several claims thinner than presented. Per-op effective-BW overrides
(extractor `_hbm_pattern` from the IR index/layout): `transpose`→`restickify` 116, `cat0`→
`stick_scatter` (size-dependent, see below), `sumcol`→`reduce_outer` 113; `add3/add4` get the
arity derate. The
false-positive fixes (`reduce_outer` requires a **kept stick dim**, excl. `sumall`;
`stick_scatter` requires a **concat dim**, excl. `transpose_outer`) are **in code but NOT in the
current records** (db_sweep predates them → still +37/+66% there); re-run needed to confirm.
Downgrades: `BW_peak=150`/`α` are a soft decomposition — only the combination is identifiable
from pointwise, and the 2:1 `add`/`mul` ratio prefers `BW_peak≈138–147` (150 imported from
read-only reductions). **`cat0` FIXED (2026-07-10, transport-shape sweep):** effBW is
SHAPE-dependent (falls with row width C, weakly R) — `252 − 4·log2R − 12.3·log2C` clamped
[45,150], R²0.93 over 10 shapes (the earlier io-based exp was WRONG: same bytes give 53–85 GB/s
by aspect). `transpose_outer` shows the SAME C-falloff (−22% at wide C) but carries no IR
pattern tag, so it stays on the default copy model, flagged — tagging it (extractor change)
would let it reuse the cat0 shape model. **Reduction large-ROWS residual FIXED (2026-07-10,
reduction-rows sweep):** the read rate falls op-independently with ROWS (149→115 GB/s over
ROWS 2048→16384, flat across COLS) — `min(150, 114+61·exp(−ROWS/3700))` on standalone
row-reductions (NOT fused softmax: gated on len(ops)==1 so input-dedup isn't broken; NOT sumcol:
reduce_outer). Reduction category 7.9%→**2.6%**. `sumcol reduce_outer=113` (rows never varied)
is still HYPOTHESIS. Arity `0.075` fit from 2 arities; per-op derate is 0.06→0.094
(superlinear); decisive test = add3/add4 with LX on. **`copy` FIXED (2026-07-09):** it is not
1R1W — `x+1.0` lowers to an `add` with a resident broadcast operand, so it is a broadcast op
(`copy`/`bcast`/`bcastcol`/`mulbcast` run ~118 GB/s vs `neg` ~105). New `bw_broadcast_gbps=118`
(applied to ops with a broadcast operand + a full input, via `_is_broadcast_op`; NOT `write`)
→ pointwise RMS 5.1→2.0%, clean broadcast points ±3%. Mechanism empirical. **`write`**
(b[1,C]+c[R,1], both broadcast → outer-product) is slow + super-linear in COLS: operands are
tiny (b=C elems, c stick-inflated R×64), so NOT an operand spill — the cost is in the
outer-product output write. No clean mechanism; modeled by an **empirical** `write_reread`
term (coef·ROWS^1.6·COLS^2.2, 12% RMS, black-box) → broadcast category 19→7.7%.
`cat0`/`cat1` are 2:1 write-heavy, not the "R=W byte-copy" the doc states.

**Matmul (k=1) — ~8% on the calibration envelope, NOT "done".** `compute(peak=1140) + two-rate
HBM + tile-spill`, with compute/HBM `overlap γ=0.46`. Isolation order: HBM (compute-free) →
compute+overlap (low cores) → spill → (K-split now gated off). The 8% holds on pow2/balanced/
mid-size shapes (~half the tested points); **honest full-range k=1 RMS ≈18% (−48…+68%), mean
bias −4%**. Both the earlier "fanout penalty" AND the "it's just tile spill" readings were incomplete.
A forced-split sweep (2 rounds, `run_split_shape_sweep.sh` + `_r2.sh`) + a 5-agent adversarial
review showed the lopsided-split miss is an **INTERACTION**: a per-core tile that is *both* large
*and* fanned out past ~8 cores (neither alone — a huge tile at low fanout is accurate; the miss
anti-correlates with weight bytes). It is size-gated and follows the LONG output dim.
**§12a split-shape term SHIPPED (2026-07-17):** `split = c·max(0, area−131072)·max(0,
log2(fan_long/8))`, `fan_long = m if M≥N else n`, `c=2.6e−3 µs/elem`, added post-overlap in
`predict_ops`. Fixes the planner-emitted tall `16×2` from −40 % → a few %, **0 on balanced/small**
(no regression). Fit + LOSO in `notes/fit_split_shape.py`. Residuals (⚠): tiny matmuls +54%
(fixed-overhead floor), forced short-dim slivers (`1×32`) & wide-`16×2` (N>M), non-pow2-N.

**DONE (2026-07-10): dropped the two-rate HBM, now a SINGLE rate = 150 (user approved).** On the
planner-realistic envelope (k=1, fanout ≤8, non-tiny, pow2-N; n=34) the old two-rate 143/156
scored RMS 7.10%; single `mm_bw_read=mm_bw_write=150` (γ unchanged at 0.46) scores **6.9%** —
equal-or-better AND physically respects the 150 copy peak. The compute-free dominant-operand
rates are ~118–148 (write corners) / ~123–136 (read corners): overlapping, both <150, no distinct
write rate. `BW_w=156` was a fit artifact absorbing overlap. **Shipped**: `mm_bw_read_gbps =
mm_bw_write_gbps = 150.0` in cost_model.py (comments/docstrings updated). Report §8 rewritten as
"adopted". Also captured:
honest matmul regime table — realistic 7.1%, forced K-split −40% (unmodeled combine, planner
avoids), skewed >8 −23%, tiny +2.5 (floor), non-pow2 −16%. γ pinned by the balanced aggregate,
NOT the small-shape GH sweep (GH alone prefers γ=0; floor-contaminated). See report §7–§12.

Adversarial review (2026-07-08) downgraded several matmul terms to HYPOTHESIS pending the re-run
- decouplers: (a) `BW_w=156 > 150` one-directional peak is physically impossible → the "compute-
free" M1 fit absorbed compute-overlap; re-fit BW_r/BW_w on the FULL model (subtract γ), not raw
bytes/time. *(Now resolved — see the RESOLVED note above.)* (b) `peak=1140 / γ=0.46` sit on a non-identifiable ridge — (1190,0.40)/(1220,0.30)
fit equal or better OOS; γ is pinned by ~one shape → need an HBM-dominant cores-scan to pin γ
alone. (c) overlap `min()` FORM untested (52/79 points cluster at compute≈HBM where all forms
coincide). (d) spill log-curve fit from ~5 pts, cap from ~2, `RB` corrupted by non-pow2 N.

**Coarse-tiling — OPEN (active).** See below.

## Coarse-tiling — softmax now largely isolated (2026-07-08)

Reframe (agreed with user): a coarse-tiled op is ONE **fused kernel** (intermediates in LX),
NOT a sum of per-op kernels. Define `rpc = ROWS/(cores·T)` = per-core rows per tile.

The `softmax_terms` grid (ROWS×T at COLS=2048) + the `coarse_terms` softmax runs (ROWS=16384 at
**both COLS=2048 and 4096**) together isolate the softmax cost, and an adversarial challenge was
run + addressed. Results (units: `us/1k-row/1k-col` ≈ per-byte cost):

- **SETTLED — the driver is `rpc`, NOT the tile count `L`.** At `rpc=16` the four points span
  `T=4..32` (4× tile count) at ~flat cost → kills any `L`/pipeline-overlap story. Normalized
  cost/row collapses onto `rpc` across 4 ROWS values.
- **SETTLED — underfill is ROWS-driven, NOT per-core-tile BYTES.** The old confound (COLS fixed →
  rows≡bytes) is broken by the cross-COLS data: at **matched `rpc`, doubling COLS (2× tile bytes)
  leaves per-byte cost unchanged (ratio 0.96–1.02)** across the whole non-spill range. So the
  underfill derate keys on `rpc` (rows), independent of COLS.
- **cost(`rpc`) is U-shaped** (per-byte): min ~40 at `rpc≈32`; steep underfill rise below
  (`rpc8`≈49, `rpc4`≈78, `rpc2`≈126 — i.e. 1.2×/1.9×/3.1× the floor); MILD rise above
  (`rpc64`≈43, `rpc128`≈46), COLS-independent (so also rows-driven, not LX pressure). The current
  derate `min(1,(rpc/16)**0.35)` is mis-shaped: under-derates `rpc≤4`, ignores the `rpc>32` rise.
- **HYPOTHESIS (leaning likely) — double-counted `arg0` read.** At the floor softmax runs at
  ~100 GB/s single-read-equiv (≈ the balanced copy rate), matching arg0-read-ONCE on both COLS;
  arg0-read-TWICE implies ~150 GB/s (at/above peak). So the fused kernel likely reads arg0 once
  (2nd read LX-served) and the model's 2× read over-counts the floor ~25%. **Confound (from the
  adversarial agent): not separated from a compute-bound (exp) floor or a BW_peak error** — the
  deciding test is a pure-copy of identical footprint vs the softmax floor.
- **SETTLED mechanism / under-sampled shape — LX-spill is BYTE-driven, separate from underfill.**
  At `rpc=256`, C4096 (2.1 MB/core tile) SPILLED (per-byte 145, `io_hbm_bytes` itself jumped as
  intermediates went to HBM) while C2048 (1.05 MB/core) did NOT. So spill triggers on per-core
  tile MB (~knee 1–2 MB/core), independent of the rows-driven underfill. Only 1 point past the
  knee → exact threshold + post-knee slope need a finer sweep.
- **Noise:** VAR (5× within-process) = 0.3%; cross-config agreement at matched rpc ≈ ±2–4%. The
  `rpc>32` "mild rise" (+7…+15%) is real vs that, but **cross-process/thermal variance is still
  unbounded** — bound it before trusting single-digit-% effects.

**IMPLEMENTED + HW-VALIDATED 2026-07-09** (cost_model.py; confirmed in the sha-`078922c` re-run):
- (a) **Fused-kernel HBM counts each distinct external input ONCE** — `_fused_hbm_bytes(ops)`
  dedups `arg`-named HBM inputs across the bundle (softmax `arg0` read by `amax`+`sub` → once).
  Fixes the ~25% floor over-count. Non-softmax ops unaffected (no `arg` reused across ops).
- (b) **Re-fit `rpc` underfill, decoupled from matmul** — new `coarse_underfill_eff` +
  `coarse_underfill_{rfull=13,exp=0.68,cap=0.95}`; matmul `pt_eff` untouched. **HW: softmax
  non-spill RMS 7.2%** (n=44, was ~20%), floor (rpc16–32) ±0.6%; residual rpc≤8 (+8–10%) and
  rpc≥64 (−7…−14%, the mild rise the cap omits) — matches the synthetic fit exactly.
- (c) **NO categorical spill term** — the plan said add one, but the IR check showed the
  extractor **already counts spilled bytes**: when a per-core tile overflows LX (~1–2 MB/core)
  the compiler moves intermediates to HBM and the IR reflects it (LX total collapses,
  `io_hbm_bytes` jumps). A predicted-knee term would double-count. The real residual is that
  spilled traffic runs slower than modeled — the HW re-run now shows the spill regime
  (rpc≥160) at RMS 24% (−18…−40%, 7 pts) vs 7.2% non-spill — a RATE effect, DEFERRED until
  the finer knee sweep gives >1 point to fit the spilled-traffic rate.

Remaining softmax decouplers: finer spill knee at C4096 `rpc∈{160,192,208,224,240,256}` (fit the
spilled-traffic rate); cross-process repeats (bound noise); cross-COLS at a 2nd ROWS. `chain`
DROPPED (per user). `matmul_row_tiling` deferred (needs `pt_eff` keyed on coarse-tile `M/tiles`).

### Decoupler sweeps — WRITTEN + design-review-vetted 2026-07-09 (added to `run_db_sweep.sh`)

Four new sweeps to upgrade the HYPOTHESIS terms; each design was adversarially challenged and
the flaws fixed BEFORE writing (memory: conservative-claims-adversarial-check). Not yet run.
- `run_pointwise_ratio_sweep.sh` — BW_peak vs α. Vetting reframed it as an explicit **read/write
  asymmetry test**: fit `R/BW_read + W/BW_write + α·min(R,W)` and CHECK BW_read==BW_write (a
  symmetric 2-param fit can never surface the misspecification the 105/138–147/150 tension hints
  at). Adds a streaming `read` probe next to the (circular) reduction anchor; sweeps ROWS for the
  plateau; `write` at small COLS is a flagged low-confidence write anchor.
- `run_matmul_gamma_sweep.sh` — peak/γ + BW_r/BW_w. Vetting confirmed the compute-dom cores-scan
  recovers peak via a **γ-independent slope** (escapes the ridge); FIXED the γ scan to a
  **spill-free small shape** (M=N=512/768, K=64, per-core tile <448) so spill can't drift into
  the γ slope. BW section is a **rank-2 (R,W) grid** with min(R,W) on both sides (the naive
  fixed-M K-sweep was BROKEN: W constant → BW_w unidentifiable, BW_r/α collinear).
- `run_nonpow2_n_sweep.sh` — the stick-padding sawtooth is in the **per-core tile N/n**, not full
  N (the naive N∈{2048..8192} step 1024 was BROKEN: all stick-aligned → sawtooth invisible; 8192
  broke the MNK cap). FIXED: forced 4×8×1, N stepped 64 so N/8 sweeps 512→576 across a stick edge.
- `run_softmax_floor_sweep.sh` — double-count vs exp-compute. Vetting rejected the untiled-copy
  control (tiling-overhead confound); added a NEW matched harness op **`softmax_noexp_row_tiling`**
  (softmax structure, `exp`→`mul`) so `T(softmax) − T(noexp)` at matched [ROWS,COLS,TILES]
  isolates exp by **wall-clock time** (not effBW, which presupposes the byte-count answer).
- `run_broadcast_sweep.sh` (2026-07-09) — pins the **broadcast effBW** (`bw_broadcast=118`,
  fit on one clean point/op) over COLS at ROWS=2048, AND confirms the **`write` spill**: a
  write ROWS×COLS grid separates C-driven (row operand `b[1,C]` spills → super-linear in C)
  from R-driven (`c[R,1]`). Report §4. `copy` is a broadcast op (increment `x+1.0`).

## Methodology (do NOT repeat past mistakes)

1. Never `measured − model_term` to isolate another term (circular).
2. Isolate each term in a regime where it DOMINATES; subtract only ALREADY-validated terms.
3. **Be conservative on every claim; before pushing a mechanism/parameter, LAUNCH adversarial
   agent(s) to challenge it** (confounds, alternatives, missing controls) and address every
   challenge, or downgrade to "hypothesis + the deciding experiment." (memory: conservative-
   claims-adversarial-check.) This caught real over-claims here (chain underfill, softmax R_eff).
4. Trust measured data, not the in-tree work_division.py model (it's a relative ranker).

## Tooling

- **Harness** `docs/source/user_guide/examples/profile_ops.py` (BENCH_OP=…; knobs BENCH_ROWS/
  COLS/N, BENCH_TILES, WD_M/N/K, SENCORES, LX_PLANNING).
- **DB rebuild** `run_db_sweep.sh` — chains all sweeps into ONE `haoyang_logs/db_sweep.log`
  (children write there via `DB_LOG`; per-run `timeout` guard) + auto-parses. New sweeps:
  `run_hbm_ops_sweep.sh`, `run_matmul_compute_sweep.sh`, `run_matmul_psum_sweep.sh`,
  `run_reread_sweep.sh` (RA/RB tile-spill, FB/FA fanout-isolation — falsified fanout),
  `run_decouple_sweep.sh`, `run_split_sweep.sh`, `run_coarse_tiling_sweep.sh`,
  `run_coarse_terms_sweep.sh`, `run_softmax_terms_sweep.sh` (active).
- **Parser** `notes/parse_sweep_logs.py` → `notes/sweep_records.{json,csv}` (merge by
  `log:lineno`, idempotent). Carries per-op split/model-term breakdown + **provenance**:
  `model_sha` (from `git:` header), `log_date`, `is_current`. Flags: `--drop-ops chain`,
  `--current-sha <sha>` (default: newest parsed log's sha). Also captures `feats` (the
  serialized `OpFeatures`) from each run's `MODEL FEATS` line.
- **Offline scorer** `notes/eval_model.py` — **recompute accuracy WITHOUT hardware** (the
  measured `kernel_us` is version-independent; only the prediction changes). The harness now
  dumps `MODEL FEATS <json>` (the model's exact input) per run for free, so a new model version
  is scored by `predict_ops(feats)` in pure Python (`cost_model.py` has no torch dep → runs
  locally). `--params k=v,...` re-scores with overridden params instantly; `--verify` checks
  feature fidelity; `--update` writes recomputed `pred_us` back. Rows lacking `feats` (the
  pre-2026-07-09 grand sweep) are reconstructed from the stored `io` block and **self-validated
  against their stored `pred_us`** (mismatches excluded; matmul needs a `feats` re-run — 119/185
  reconstruct today). THE model-iteration loop: edit params/form → `eval_model.py` → new
  accuracy, no Spyre. (`cost_model.op_to_dict`/`op_from_dict`/`ops_to_json` do the (de)serialize.)
- **Extractor** `dump_cost_model.py`: `_matmul_features` (MACs, M/m, N/n, |A|, |B|, k),
  `_hbm_pattern` (restickify / stick_scatter / reduce_outer from IR index+layout).

## Immediate next steps (2026-07-08 vintage) — ⚠️ SUPERSEDED, see CURRENT STATE at the top

> **Do not act on this list.** It predates the bmm layout work, the close-out sweeps and the
> scope decisions, and several items are now known to be resolved or wrong. Item 0's "batch floor"
> was in fact largely a **device tensor layout** effect (report §13); items 1–3 have been run.
> Retained only as the record of the plan at that date.

0. **bmm (current focus).** §12a split-shape term is shipped for mm. bmm is NOT just mm×batches:
   a forced `b=1` (batch serialized) sweep shows a large **batch floor** — bmm is −78 % even at a
   *balanced* `4×8` split. Isolation order (agreed): **(a) bmm batch floor** (the dominant miss,
   likely per-batch weight reload / pipeline drain — isolate at `b=1`, then the shared-weight 3d2d
   control) → **(b) bmm split-shape** (should inherit the §12a mm term on top) → **(c)** the forced
   `b=B` batch-split pathology (−90 %; the planner never emits it → a guard/warning, not a term).
1. **Re-run `run_db_sweep.sh` on current code** (psum gate + extractor fixes) → the ONE clean
   current-model dataset. Master runner auto-stamps this sha as `is_current` and drops chain.
   Everything below depends on having current-model `pred_us`.
2. **softmax_terms sweep** (running) → VAR (is the ~19% swing real?) → GR (L vs rows/tile) →
   SP (spill knee). Then adversarial-challenge the conclusion BEFORE modeling.
3. **Decoupler sweeps** the adversarial review proved necessary (fold into the re-run) — for the
   report's "hypothesis → isolation" narrative: pointwise write-only + read-only probes (break
   the `BW_peak`/`α` degeneracy); add3/add4 with LX on (arity mechanism); `cat0` size/aspect +
   `sumcol` reduced-dim (rows) sweeps; matmul HBM-dominant cores-scan (pin γ alone) + BW_r/BW_w
   re-fit on the full model; non-pow2-N handling.
4. If a real, isolable coarse driver: fused-kernel HBM (count reused inputs once) + the
   L-or-rows/tile term + the categorical LX-spill. Re-verify via db_sweep.
5. `matmul_row_tiling` pt_eff keyed on coarse-tile M; recheck.
