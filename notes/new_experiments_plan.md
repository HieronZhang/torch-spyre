# New experiments — comprehensive plan (post-report review)

Design doc for the next hardware sweep. It collects **every open item the report currently flags**
(hypothesis / queued / "we don't have the control" / deciding-experiment) and turns each into a
concrete, isolable experiment.

Each entry has: **goal** (the open item + section), **isolation** (the control that makes it clean),
**config** (`BENCH_OP` + shapes + knobs), **expected** (what would confirm vs refute), **infra**
(existing op / new op / new extractor feature), **priority**.

Methodology (from `cost_model_status.md`, do not repeat past mistakes):

1. Isolate each term in a regime where it **dominates**; subtract only **already-validated** terms —
   never `measured − model_term` to back out another term (circular).
2. Fit on **absolute-µs residual**, not err%, where err% is confounded by a size gradient.
3. **Adversarial-review every mechanism/parameter before it is written as settled** — challenge with
   confounds/alternatives/missing-controls, or downgrade to "hypothesis + deciding experiment."
4. Trust measured data, not the in-tree `work_division.py` ranker.

Harness: `docs/source/user_guide/examples/profile_ops.py` (`BENCH_OP`, `BENCH_ROWS/COLS/N/B`,
`BENCH_TILES`, `WD_B/M/N/K`, `SENCORES`, `LX_PLANNING`). All of the experiments below are packaged in
one runnable script — **`run_new_experiments_sweep.sh`** (the new comprehensive overnight sweep, which
**supersedes the stale `run_overnight_sweep.sh`**). It self-parses into `notes/sweep_records.json`;
score offline with `eval_model.py` — no hardware needed once the log returns. Sections map 1:1 to the
IDs below: `GAMMA`(X1) `BWCORES`(X2) `ARITY`(P1) `THINK`(M1) `TINY`(M2) `MSPLIT`(M3) `BMM`(M4)
`COARSE`(C1/C2). Run a subset with `SECTIONS="GAMMA BWCORES" bash …`.

```bash
bash docs/source/user_guide/examples/run_new_experiments_sweep.sh   # ~143 runs, ~2.5–5 h
```

---

## Tier 0 — foundational (the whole model rests on these)

### X1. γ(cores): the compute/memory overlap fraction vs core count
- **Goal (§10).** The model uses a single `γ = 0.46`; fig 10 shows the lower-core runs mispredicted and
  the note flags an `L`-dependent γ. The report already measures the hidden fraction falling
  `0.64 → 0.49 → 0.35` as cores go `4 → 8 → 32` at one shape — this generalizes it.
- **Isolation.** ONE fixed matmul shape where compute ≈ memory (so overlap is visible), vary **only**
  the core count.
- **Config.** `BENCH_OP=mmwd`, `M=K=N=2048`, forced splits summing to each core count:
  `2×2×1` (4), `2×4×1` (8), `4×4×1` (16), `4×8×1` (32); repeat at `4096×2048×2048` and
  `2048×4096×2048`. Cross-check with `BENCH_OP=mm` + `SENCORES ∈ {1,2,4,8,16,32}`.
- **Expected.** Hidden fraction (from `measured` vs `additive` and `overlap` predictions) falls with
  cores → fit `γ(L)` (a small power law or table). Refutes the single-γ if the fit is flat.
- **Infra.** Existing (`mmwd` + `WD`, or `mm` + `SENCORES`). **Priority: HIGH.**

### X2. cores → effective-BW for a memory-bound op
- **Goal (§16).** `softmax_unrolled` is 91 % off because every row is `cores=1` and the memory term has
  no per-core BW scaling. The whole memory model assumes ≥ 2 cores; this calibrates `BW(cores)` and
  validates that assumption.
- **Isolation.** A pure memory-bound op (no compute), vary only cores.
- **Config.** `BENCH_OP ∈ {neg, read, softmax_row_tiling}`, fixed large shape `8192×2048` (and
  `16384×4096`), `SENCORES ∈ {1,2,4,8,16,32}`.
- **Expected.** effBW rises from ~6–7 GB/s @ 1 core to a ~150 plateau; fit `BW(cores)`. Fixes
  `softmax_unrolled` properly and confirms the ≥ 2-core memory model.
- **Infra.** Existing + `SENCORES`. **Priority: HIGH.**

---

## Tier 1 — open modeling questions

### P1. §3 dependent add-chain: read-after-write vs bundling (the deciding experiment)
- **Goal (§3).** `add_k` runs ~7.5 % / round-tripped intermediate above the same bytes with no
  dependency. We attributed it to the read-after-write on `buf0`, but cannot yet separate that from a
  single-fused-kernel scheduling effect — we lack the controls.
- **Isolation.** Three **matched-byte** variants (all 4R:2W at `k=3`):
  - `add3` — dependent chain (`op1` reads `buf0`). [have it]
  - `add_indep2` — ONE kernel, TWO **independent** adds (`op0=a+b`, `op1=c+d`): same 4R:2W, **no**
    dependency. [NEW op] → `add3 − add_indep2` = the pure read-after-write cost.
  - `add3_sep` — the same `a+b+c` forced into TWO **separate** kernels (graph break). [NEW op] →
    `add3 − add3_sep` = the bundling cost.
  All three under `LX_PLANNING ∈ {0,1}` (on → intermediate on-chip; the report's corroboration).
- **Config.** shapes `2048×{1024,4096,16384}`. Extend arity `add5`, `add6` to confirm the
  per-intermediate cost stays linear (~7.5 % each).
- **Expected.** `add3 ≈ add_indep2` → cost is bundling, not RAW (surprising). `add3 > add_indep2` →
  RAW round-trip confirmed (expected). `add3 ≈ add3_sep` → bundling is free.
- **Infra.** NEW ops (`add_indep2`, `add3_sep`); `add5/add6` = trivial `_NARY` extend. **Priority: MED-HIGH.**

### M1. Tensor-size extreme: the thin-K residual
- **Goal (§12).** `K ≤ 128` matmuls are +10…+32 % (thin-operand memory/tile residual, `n=6`).
- **Isolation.** Vary **only** `K` into the thin regime at fixed large `M=N`, balanced split.
- **Config.** `BENCH_OP=mmwd`, `M=N=2048`, `K ∈ {16,32,64,128,256,512}`, `WD 4×8×1` (32 cores);
  second row at `M=N=4096`.
- **Expected.** Residual grows as `K → 0` (compute vanishes, a memory/tile floor dominates) → decide
  whether it is a fixed per-tile floor or a thin-operand BW derate; fit on absolute-µs.
- **Infra.** Existing. **Priority: MED-HIGH.**

### M2. Tiny: the small-output floor
- **Goal (§12).** `min(M,N) ≤ 512` matmuls are +8 % mean, up to +48 % — a suspected fixed per-kernel
  floor.
- **Isolation.** Shrink `M,N` at fixed `K`; is the excess fixed-µs (→ per-kernel) or per-byte?
- **Config.** `BENCH_OP=mmwd`, `K=2048`, `M=N ∈ {128,256,512,1024,2048}`, `WD 4×8×1`.
- **Expected.** Constant-µs excess → add a per-kernel matmul floor; shrinks in % with size → already
  captured by the existing terms.
- **Infra.** Existing. **Priority: MED.**

### M3. Deep-tail extreme splits (refine the §12 knees)
- **Goal (§12).** `32×1` over-predicts (+22 %), `1×32` under (−16 %) — the two-sided knees do not
  fully reach the deep tail; today constrained by ~one base shape.
- **Config.** `BENCH_OP=mmwd` at `2048×4096×2048`, `6144×2048×2048`, `4096×4096×2048`; splits
  `{32×1, 1×32, 16×2, 2×16, 8×4}` — each of `fan_long`, `fan_short` seen at ≥ 3 values.
- **Expected.** Refit `c_L`, `c_S` and the knees on the wider set; check the 32×1 over-prediction is
  not a systematic long-dim over-charge.
- **Infra.** Existing. **Priority: MED.**

### M4. §13 bmm two-rate model — validate and decide ship
- **Goal (§13).** The two-rate effective-BW model (`BW_stream≈64`, `BW_reread≈16`) hits 36 % offline
  but is not in the code; the rate drifts ~2× with shape (the open item).
- **Isolation.** full bmm vs `3d2d` (shared weight) across `B` and `K` at matched split — the
  full-minus-3d2d delta is the per-batch weight re-read `(B−1)·K·N·2`.
- **Config.** `BENCH_OP ∈ {bmm_wd, bmm_wd_3d2d}`, balanced `4×8`, `b=1`, `B ∈ {2,4,8,16,32}`, sweep
  `K ∈ {256,1024,4096}` and a couple of `M,N`; plus the forced-`b=B` guard case.
- **Extractor.** Expose per-batch reread bytes in `_matmul_features` / `OpFeatures` (the term needs it).
- **Expected.** Two-rate model tracks full vs 3d2d across the fuller set → refit the two BWs, decide
  ship vs keep-documented; or a shape-dependent rate emerges → model that instead.
- **Infra.** Existing ops + NEW extractor feature. **Priority: HIGH.**

---

## Tier 2 — coarse tiling

### C1. `matmul_row_tiling` per-tile underfill (tile ladder)
- **Goal (§16).** Kernel grows with tile count (386→652 µs, tiles 1→8) while `pt_eff=1`; two rows with
  **identical extracted features measure 29 % apart** → unfittable until the extractor distinguishes them.
- **Config.** `BENCH_OP=matmul_row_tiling`, `BENCH_TILES ∈ {1,2,4,8,16,32}` × 3–4 shapes varying `M`
  (per-tile `M = M/tiles`) and `N`.
- **Extractor (prereq).** Emit a per-tile-M / split-arrangement feature so the identical-feature pair
  separates. Keep `K` whole, fanout ≤ 8 (no §12), so the new term is fit on the residual after §11/§15/§16.
- **Priority: MED (needs the extractor feature first).**

### C2. Nested validation (after the code fixes)
- **Goal (§ coarse table).** `mm_nested_m_k` (−38 %, missing `×loop_trip`) and `bmm_nested_b_k` (+15 %,
  fractional-trpc) are known **code defects** — fix and re-score existing rows, then confirm with a
  small sweep.
- **Config.** After the fixes: `mm_nested_m_k` and `bmm_nested_b_k` vs `matmul_k_tiling` (flat K-tiling,
  5–7 % accurate control) at **matched** shape and total tile count — the delta is the outer nest.
- **Priority: MED (code-first; mostly a re-score, small confirming sweep).**

### C3. 2-D coarse tiling
- **Goal (plan C2).** Tile **two** output dims at once — not in the suite; do the per-tile / nested
  terms compose?
- **Config.** NEW op variant (tile `M` and `N` together), small grid.
- **Priority: LOW.**

### C4. sencores dependence of the coarse terms
- **Goal (plan C3).** `eff` and `s_lx` were fit at 32 cores; `softmax_unrolled` forces `cores=1`. Do the
  terms hold off the 32-core point? (overlaps X2.)
- **Config.** A coarse op (`softmax_row_tiling`) at `SENCORES ∈ {1,8,32}`, fixed shape/tiles.
- **Priority: LOW-MED.**

---

## Part I secondary (mechanisms flagged but low-stakes)

### P2. §4 broadcast-operand effBW
- **Goal.** A broadcast operand runs ~118 GB/s vs plain 1R:1W ~105 (mechanism open, ~10 % residual).
- **Config.** `bcast`/`bcastcol`/`write` at varying broadcast-dim size and pattern, fixed total bytes.
- **Priority: LOW-MED.**

### P3. §5 `write` outer-product re-read
- **Goal.** The `write` op's empirical super-linear extra bytes (mechanism open).
- **Config.** `write` at varying output aspect `R:C`, fixed area.
- **Priority: LOW.**

---

## Status / what is built

- **`run_new_experiments_sweep.sh`** — the comprehensive new overnight sweep, all eight sections
  above, self-parsing. **Ready to run.**
- **`profile_ops.py`** — new ops for P1: `add5`, `add6` (arity ladder) and `add_indep2` (the
  independent-adds control). **Done.**
- **Pending infra (not blocking the run):**
  - M4 ship: expose per-batch reread bytes in `dump_cost_model.py::_matmul_features` /
    `OpFeatures` — only needed to *ship* the two-rate term; the `BMM` section still gives the data.
  - C1: an extractor feature distinguishing the identical-feature `matmul_row` pair (fit-blocked
    until then; the `COARSE` tile ladder still measures the effect).
  - C2: the coarse code fixes (`mm_nested ×loop_trip`, `bmm_nested` trpc clamp) — a no-HW re-score.

## Suggested reading order once the log returns

1. **GAMMA + BWCORES** (foundational: γ(cores), BW(cores)) — refit these first; every downstream term
   inherits them.
2. **THINK + TINY + MSPLIT** (the flagged matmul residuals).
3. **ARITY** (the §3 deciding experiment: `add3 − add_indep2` = the read-after-write cost; LX on/off).
4. **BMM** (refit the two-rate BWs on the fuller set → decide ship vs keep-documented).
5. **COARSE** after the code fixes land.

No new hardware capability is required — only run time.

---

## Next model — a byte-keyed read-after-write term (and unifying it with coarse-tiling spill)

**What §3 established.** The margin on a chained/fused op sequence is a **read-after-write dependency**
across op boundaries, *not* fusion (fused ≈ separate at 1, 3, 4 dependent reads). It is a real,
first-class cost a per-kernel byte model cannot see, and it is **gated off when the intermediate is
on-chip** (scratchpad on → the model already predicts it within a few %).

**Proposed form — additive, keyed on the intermediate's size (not a `×derate` on `m`):**

```text
T = byte_model  +  Σ_intermediates  (round-tripped_bytes) · c_raw        # HBM-resident intermediates only
```

Three reasons this is the right shape:

1. **Byte-proportional, and the data agrees.** The §3 excess in `add`-units is **shape-invariant**
   (≈ 0.13 per dependent read at COLS 1024/4096/16384) — the cost scales with the intermediate size,
   which an additive byte term captures and a fixed-% derate does not.
2. **Composes with the LX model for free.** The term is present only for HBM-resident intermediates; on
   chip there are no round-tripped bytes, so it is automatically 0 — matching the validated scratchpad-on
   predictions. A `×(1+0.075·(m−1))` derate has to special-case that.
3. **Generalizes past adds.** A read-after-write between *any* two dependent ops (matmul→bias,
   softmax→dropout, …) is the same phenomenon; `m`-counting only works for a homogeneous add chain.

**The key connection — this is the SAME thing as coarse-tiling §15 (`s_lx`).** A coarse-tiled kernel
(softmax = 5 fused ops, matmul_row, …) IS a multi-op fused chain. While its intermediates fit LX they
stay on-chip → no read-after-write cost (the model correctly adds nothing). When the per-core working
set overflows LX they **spill to HBM**, and each spilled intermediate is *written by one fused op and
read by the next* — a read-after-write round-trip, exactly §3's mechanism. Today §15 charges this as an
empirical **BW derate** on the spilled bytes (`BW *= (512KB/ws)^0.15`, softmax-calibrated); §3 charges
the add chain as an **arity derate**. **These are two parameterizations of one physical effect.** A
single additive read-after-write term keyed on the round-tripped intermediate bytes could **replace
both** and be physically consistent across pointwise chains, coarse tiling, and program-level op
sequences.

**Should we apply it to `add3`/`add4`/`add5` now? — no, hold off.** The add-chain data is too irregular
to fit any simple form cleanly: the `add4` anomaly inflates depth 2, and neither the multiplicative
derate (over-predicts `add5`/`add6`) nor a linear additive term (under-predicts `add4`–`add6`) matches
the whole ladder. The shipped `×(1+0.075·(m−1))` happens to fit the realistic `add3`/`add4` and is kept
as a flagged placeholder. Fitting `c_raw` on the pathological add chain would encode the anomaly.

**Where to develop and fit it instead — coarse tiling.** Coarse-tiled kernels give genuinely multi-op
chains with intermediate sizes that vary cleanly (via tile count and COLS), no `add4`-style pathology,
and a spill regime that already isolates the HBM read-after-write. Plan:

- Re-derive §15 as `T += spilled_intermediate_bytes · c_raw` (additive) instead of a BW multiplier, and
  refit on the existing `softmax_row_tiling` spill sweep; check it does not regress the 5.7 % RMS.
- Use §3's clean `add3` anchor (≈ 0.13 `add`-units per read, shape-invariant) as an independent sanity
  check on the fitted `c_raw` magnitude — the two should be within a small factor if it is one effect.
- Only then consider a **program-level** read-after-write term for arbitrary op sequences.

**Detection is easy — the model already sees the chain.** A multi-op fused bundle carries multiple
ops in its `feats` (e.g. `add3` = 2 ops, `add4` = 3, softmax = 5), while a single op has 1. So the model
can *identify* a dependent chain today; what is missing is (a) the term to charge it and (b) which
intermediates are HBM-resident (cross-op read-after-write). This is the future fix: apply the byte-keyed
read-after-write term to bundles with >1 op whose intermediate spills HBM, instead of leaving them
under-predicted.

**Coverage gap to fix first — we have only ONE multi-op coarse example.** Of the coarse-tiled
workloads, only the **softmax family** (`softmax_row_tiling` / `_noexp` / `_unrolled`, all 5-op) is a
genuine multi-op fused chain; every other coarse op (`matmul_row`, `matmul_k`, `mm_nested`, `bmm_*`,
`ct*`) is **single-op** (1 op in the bundle, no cross-op read-after-write). Fitting `c_raw` on softmax
alone repeats the one-op-family problem. So a prerequisite is **new coarse multi-op chains** whose
intermediate count and size vary cleanly — candidates: **LayerNorm** (mean/sub/var/rsqrt/mul), a
**coarse-tiled pointwise chain** (e.g. `((a+b)*c+d)` tiled, so #intermediates is a knob and the tile
size sets spill), and a **GELU/normalization chain**. These give the independent multi-op examples the
unified read-after-write term must be fit and validated against.

**Prerequisites** (same as before): op-*sequence* data beyond chained adds; the extractor exposing
cross-op read-after-write dependencies (it sees within-bundle ops only today); and understanding or
bounding the `add4` anomaly so it does not contaminate the fit.
