# Spyre Cost Model — Status / Handoff (updated 2026-07-24)

Living status + detail doc — **read this first on resume**. The full model write-up is
[cost_model_report.md](cost_model_report.md) (the MAIN doc — the long-form derivation of every
term, with figures and accuracy). This file holds the details that don't belong there:
implementation state, open work, methodology, tooling, next steps. Keep the two in sync — if a
number changes here, update the report.

Goal: a HIGH-LEVEL **relative** cost model over the after-pre-scheduling LoopLevel IR, to
guide optimization (LX placement, coarse-tiling). Bar: correct ranking + ±15-20%. No local HW —
edit locally, hand commands to a run machine, logs pasted back.

---

## ⭐ CURRENT STATE & PLAN (2026-07-24) — supersedes the older dated sections below

**The active effort is the "outlier attack": fix every ≥10% mispredicted *normal/valid* input,
mechanistically and noise-controlled** (standing directive, see memory `outlier-attack-six-categories`;
detailed plan `~/.claude/plans/shimmering-percolating-rivest.md`). Ignore tiny matmuls and
1×32/32×1 extremes.

### 🤖 AUTONOMOUS RUN IN PROGRESS (2026-07-24, session 2) — cats 3→6 then flash-attn

Running unattended (~4h). A 5-min heartbeat cron re-invokes and reads THIS section to continue.
Discipline (MUST): mechanism-based modeling; **adversarially review EVERY claim with a Workflow of
challenger agents before acting** (user mandate, repeated); design isolation sweeps + hand off if
data is thin (don't wait); dump lower-level IR locally if mechanism unclear; goal <10%/point; never
commit/git; regenerate figure + full end-of-section table after any model change; keep report+status
consistent. Analysis scripts: `notes/analyze_matmul_overlap.py` (cat3), `notes/analyze_bmm_layout.py`
(cat4). **Findings below are UNDER REVIEW / not yet verified until their challenge workflow passes.**

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
1. `run_transport_iso_sweep.sh` — cat 2 (ALREADY RAN, folded; transport done at 5.5%).
2. **`run_matmul_family_sweep.sh`** — cats 3+4+5+6 in ONE run (~133 runs): MMISO_CORE (forced
   1/2/4/8/16/32-core matmul = clean γ / low-core rate), MMISO_SPLIT, MMISO_BATCH (bmm), CTFILL
   (coarse per-tile fill/drain), BWCORES (BW-vs-cores for reductions/softmax). This is the ONE sweep
   to run for the whole matmul-family Part III rework. (Supersedes the earlier separate
   run_matmul_overlap_iso_sweep.sh + run_coarse_bwcores_sweep.sh, now merged.)
3. `run_flash_resweep.sh` — cat 7 (fixed flash configs + IR).
Run #2 → then the coordinated Part III rework (cats 3+4+6) + cat-5 BW(cores) can be fit + shipped on
clean data. Report-ready prose for the Part III rewrite is pre-drafted in `notes/part_iii_rewrite_draft.md`.

### Data + how to score (do this on resume)

- **Overnight sweep** `haoyang_logs/outlier_20260723_072217.log` (450 usable points) + targeted
  **broadcast small-ROWS sweep** `haoyang_logs/bcast_smallr_20260724_015428.log`, both folded into
  `notes/sweep_records.json`. Per-run IR dumps in `haoyang_logs/ir/`.
- **Noise protocol** now in `profile_ops.py`: `BENCH_REPS` back-to-back profiled measurements →
  `kernel_us_min/median/std/cv` in the SUMMARY. **Score on CLEAN, MATCHED data** (cores=32, low
  `cv`). `eval_model.py --all` mixes core counts (a real `BW(cores)` effect) and old runs, so it
  is NOT the accuracy number — filter to matched conditions.
- Sweep scripts: `docs/source/user_guide/examples/run_outlier_sweep.sh` (the superset, 9 sections,
  budget-guarded via `MAX_SECONDS`), `run_broadcast_smallr_sweep.sh`.

### The six categories — status + key finding

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

### Immediate next step

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

## Immediate next steps

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
