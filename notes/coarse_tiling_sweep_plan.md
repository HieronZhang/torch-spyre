# Coarse-tiling modeling — detailed sweep plan (draft for review)

The split-shape term (§12a) closed the lopsided-matmul miss. The remaining large residuals are the
**coarse-tiled** matmul/softmax categories. Current accuracy (eval_model, current model):

| category | RMS | signal from existing data |
|---|---|---|
| `matmul_row_tiling` | 22 % | kernel **grows with tile count** (386→652 µs, tiles 1→8, M=N=K=2048) while pred is ~flat (~360) → a **per-tile overhead** the model omits. A dip at tiles=2 (337 µs) = LX reuse. |
| `mm_nested_m_k` | 35 % | kernel **explodes on nesting** (390→1113→1723→2828 µs, tiles 1→8) while pred crawls (393→1642) → a **~−40 % flat nested overhead**. |
| `softmax_unrolled` | 91 % | pred is **~10× too small** (21 µs vs 287 µs, [1024,512]) and ~flat in tiles → almost certainly the **extractor under-counts** the unrolled loop's work (no `CoarseTileInfo`, `sencores=1`), a feature bug not a model gap. |
| (`matmul_k_tiling` 5.7 %, `coarse_reduction` 3.6 %, `softmax` 5.7 % — GOOD, controls) | | |

**Guiding principle: separate an EXTRACTION bug (features wrong → pred garbage) from a MODELING gap
(features right, term missing).** The fix and the experiment differ. `softmax_unrolled` pred≈0
looks like extraction; `matmul_row`/`mm_nested` grow monotonically → likely modeling. Confirm each
by inspecting the stored `feats` (loop_trip, tiles_output_dim, bytes, macs) BEFORE fitting anything.

---

## Phase A — Diagnose (partly doable NOW from stored feats; no HW)

For each broken op, dump the `feats` of existing rows and check:

- **A1 `softmax_unrolled`.** Does `loop_trip == tiles`? Do the summed HBM bytes ≈ the true
  `2·B·D·2B`? Is `tiles_output_dim` set? If pred≈21 µs because the bundle only carries **one tile's**
  bytes (or `loop_trip=1`), it is an **extraction bug** in the unrolled path (`dim_hints`, no
  `CoarseTileInfo`) — fix `dump_cost_model.py`, then the existing sweep already validates it.
- **A2 `mm_nested_m_k`.** Does the feats bundle count the **outer M×2 nest**? Is `loop_trip` the
  product of both nest levels, or only the inner K tiles? A −40 % flat miss that appears the moment
  nesting starts smells like the outer loop's re-reads (or its `loop_trip`) being dropped.
- **A3 `matmul_row`.** Feats should be right (it uses `CoarseTileInfo`); the growth-with-tiles is a
  genuine per-tile cost. Confirm `loop_trip==tiles` and the bytes scale so we know it is modeling.

Deliverable: for each op, "extraction bug (fix code, re-validate on existing data)" vs "modeling gap
(sweep + fit)". This gates whether a sweep is even needed.

---

## Phase B — Targeted sweeps (only for confirmed MODELING gaps)

### B1. `matmul_row_tiling` — the per-tile overhead (§16, queued)
The base model sets a coarse-tiled matmul's `pt_eff=1` (per-tile systolic underfill flagged, not
modeled). Kernel grows with tiles; we need the per-tile cost as a function of the per-tile M.

- **Fine tile ladder × shapes:** tiles {1,2,4,8,16,32} at 3–4 shapes varying M (per-tile M = M/tiles)
  and N (output width). Goal: does the excess grow with tile count, or with per-tile M (rows/tile),
  or with #tiles×fixed-overhead?
- **Isolate the tiles=2 LX-reuse dip from the overhead climb:** hold the working set below LX (small
  per-tile) so the reuse benefit is constant, and separately grow it — so the U-shape (reuse dip then
  overhead climb) is decomposed.
- **Confound to control:** as tiles↑ the per-core tile shrinks → §11 spill and §15 `eff` (coarse
  memory underfill) already move. The new term must be fit on the residual AFTER those, and checked
  not to double-count. Keep K whole (no K-split) and fanout ≤8 (no §12a).

### B2. `mm_nested_m_k` — the nested overhead
Nesting (outer M×2, inner K×tiles) adds a flat ~−40 %. Decompose:

- **Nested vs flat control:** `mm_nested_m_k` vs `matmul_k_tiling` (flat K-tiling, 5.7 % accurate) at
  **matched** shape and matched total tile count → the delta is purely the outer-M nest.
- **Vary the nest independently:** sweep the **outer** M-split count and the **inner** K-tile count
  separately (needs a builder knob for the outer count; today it is hard-coded ×2). If only a builder
  knob is missing, note it.
- **Is it re-read or loop overhead?** The outer nest re-streams the shared operand per outer step; a
  re-read term scales with operand bytes × outer count, a loop-overhead term is fixed per step.
  Sweep shape (operand size) at fixed nest to separate.

### B3. `softmax_unrolled` — validate after the A1 extraction fix
If A1 is an extraction bug, fix it and re-score the EXISTING rows (no HW). Then a small confirming
sweep:

- **Unrolled vs looped control at matched [B,D,tiles,sencores=1]:** `softmax_unrolled` vs
  `softmax_row_tiling` (looped) at single core. If, post-fix, looped and unrolled agree, done.
- **Scaling:** vary B (the unrolled trip count), D, tiles — confirm the corrected pred tracks kernel.

---

## Phase C — Broader coverage (potential gaps not yet probed)

- **C1 Nested is systematic.** Both `mm_nested_m_k` (−40 %) and `bmm_nested_b_k` (−64 %) are nested and
  both badly off. Whatever B2 finds should be checked to also explain the bmm nested case (shared
  mechanism → one term).
- **C2 2-D coarse tiling** (tile two output dims at once): not in the suite at all. A real compiler
  may emit it; probe a small grid to see if the per-tile/nested terms compose.
- **C3 `sencores` dependence.** `softmax_unrolled` forces `sencores=1`; the coarse terms (`eff`,
  `s_lx`) were fit at 32 cores. Sweep a coarse op at `sencores ∈ {1,8,32}` to check the terms hold
  off the 32-core point.
- **C4 LX-spill interaction (§14 `s_lx`).** The per-tile working set crosses the ~512 KB LX knee as
  tiles change; make sure B1/B2's new term is separated from `s_lx`, and that `s_lx` is right for the
  matmul coarse path (today it is gated to non-matmul).
- **C5 Larger tile counts / very small per-tile M** to find where per-tile underfill saturates.

---

## REVIEW VERDICT (adversarial, verified against stored feats + code) — supersedes the above

The review re-scored every row from stored `feats` on the current model and diagnosed each miss.
**Most of these are code fixes re-scoreable on existing data, NOT sweeps.** Corrected picture:

- **`softmax_unrolled` (91 %) — MODELING gap, needs NEW data.** feats are CORRECT (`loop_trip==tiles`,
  bytes right). The real cause: **every row is `cores=1`, and the memory term has no per-core BW
  scaling** — measured effective BW at 1 core is **6–7 GB/s** vs the model's assumed ~100 (the same
  softmax at 32 cores runs 82–98 GB/s and is modeled to <10 %). Fix = a cores→BW factor, but it
  **cannot be calibrated from existing data** (cores=1 non-matmul is a singleton, confounded with the
  unrolled path). → the one genuinely-missing **HW sweep**: effective BW vs core count.
- **`mm_nested_m_k` (35 %) — CODE DEFECT, no sweep.** `predict_ops` charges `matmul_macs/cores`
  **without `*loop_trip`** ([cost_model.py:740](../torch_spyre/_inductor/cost_model.py)), so only one
  tile's compute is counted. Fix + re-score. Caveat: `matmul_macs` is **per-tile** for mm_nested but
  **full-tensor** for matmul_row (extractor inconsistency) — the fix must handle both.
- **`bmm_nested_b_k` — CODE DEFECT (opposite sign), no sweep.** −79 %@tiles=1 then **+48/+87 %**
  (over-predict): the extractor emits **fractional `tile_rows_per_core`** (batch over-split) →
  `coarse_underfill_eff` crashes → mem explodes. Distinct from mm_nested → **NOT one nested term** (C1
  refuted). Clamp/fix the batch-split trpc and re-score.
- **`matmul_row_tiling` (22 %) — genuine gap, but feature-limited.** `pt_eff` is force-set to 1.0 for
  tiled matmuls; residual keys on rows/core. **First** try re-enabling `underfill_eff` and re-score.
  But two rows with **identical extracted features measure 29 % apart** → the split *arrangement* is
  not in the feature vector; a sweep is unfittable until the extractor emits a distinguishing feature.

### The plan now (code-first)
1. **Code fixes, re-score existing rows (NO HW):** (a) mm_nested compute `*loop_trip` (reconcile the
   per-tile-vs-full `matmul_macs` inconsistency first); (b) bmm_nested fractional-trpc clamp; (c)
   matmul_row re-enable `underfill_eff`. These likely close mm_nested and bmm_nested and shrink matmul_row.
2. **The ONE HW sweep (in the overnight batch):** **cores→effective-BW calibration** — a memory-bound
   non-matmul op (pointwise, reduction, looped softmax) swept over `SENCORES ∈ {1,2,4,8,16,32}` at a
   couple of shapes. Fixes softmax_unrolled properly and validates the ≥2-core assumption the whole
   memory model rests on.
3. **Deferred (needs an extractor feature first):** a matmul_row tile ladder — only if the code fix
   leaves it unresolved, and only after the extractor distinguishes the identical-feature pair.

Dropped from the overnight batch: the CR/CN/SU re-runs — they re-confirm known code bugs, not new data.
