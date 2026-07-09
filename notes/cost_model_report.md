# Deriving a Cost Model for the IBM Spyre Accelerator

*Working draft. This is the long-form DERIVATION: how each term of the model was arrived
at, starting from an observation in the sweep data, then a question, a hypothesis, an
isolation experiment, and the resulting model form — validated offline with
`notes/eval_model.py` against the stored measured times (no re-run needed to re-score).
Every section follows that same arc. The concise summary lives in
[cost_model_presentation.md](cost_model_presentation.md).*

*Section skeleton — filled in iteratively, one section at a time; figures are generated
by `notes/plot_report.py` from `sweep_records.json`.*

---

## Part I — Pointwise: the gold baseline

### §1. Pointwise kernels are memory-I/O bound

- **Observation:** kernel time grows linearly with the number of bytes moved, straight
  through the origin (no fixed per-kernel cost).
- **Model (baseline):** `T = bytes / BW`. No compute term. This is the reference every
  later op is measured against.
- **Figure:** measured time vs device bytes for `neg`/`gelu`/`exp`; linear fit, intercept ≈ 0.

### §2. The read/write ratio changes the effective bandwidth

- **Observation:** the effective BW (`bytes/time`) is NOT constant — a balanced 1R:1W op
  (`neg`) runs *much slower* per byte than a read-only op (a reduction), with 2R:1W
  (`add`) in between. So `bytes/BW` with a single BW is wrong.
- **Question:** what makes balanced traffic slower than one-directional traffic?
- **Hypothesis:** HBM is a shared bus that pays a **turnaround** cost when it switches
  between reading and writing; the penalty falls on the overlap `min(R,W)` →
  `T = (R+W)/BW_peak + α·min(R,W)`, giving the V-shaped effective BW.
- **Experiment:** `run_pointwise_ratio_sweep.sh` — effBW across R:W ratios. Framed as an
  **asymmetry test**: fit `R/BW_read + W/BW_write + α·min(R,W)` and check whether
  `BW_read == BW_write` (rather than assuming one symmetric `BW_peak`).
- **Figure:** effBW vs R:W ratio (the V-curve), model overlaid.
- **Open:** whether `BW_peak/α` is a clean symmetric decomposition — the 2:1 data hints
  `BW_peak ≈ 138–147`, not the read-only 150. The ratio sweep settles this.

### §3. Chained pointwise ops pay a per-op derate

- **Observation:** `add3`/`add4` (2/3 chained adds) run slower per byte than a single
  `add`, and the slowdown grows with the number of ops.
- **Question:** is that extra cost the intermediates round-tripping HBM (already in the
  byte count), or a separate per-op effect?
- **Hypothesis:** a multi-pass chain pays `×(1 + 0.075·(n_ops − 1))` on top of its bytes.
- **Experiment:** arity × size, and the deciding **LX-on vs LX-off** run — if the derate
  survives when intermediates stay in LX (no HBM round-trip), the mechanism is not the
  round-trip.
- **Status:** HYPOTHESIS (fit on 2 arities); mechanism test pending.

---

## Part II — Other memory-bound ops

### §4. Broadcast: a broadcast operand is loaded once, not per element

- **Observation:** `bcast` (`a[R,C] + b[1,C]`) costs about the same as a single streaming
  pass over `a` — NOT two full reads. The small operand `b` does not scale with the output.
- **Model:** count each broadcast operand at its own (one-row/-col) device size, loaded
  once; it is a real but tiny load, not zero and not output-sized.
- **Residual (open):** `write` (both operands broadcast → near pure output) degrades
  **super-linearly in C** — a turnaround term on `min(R,W)` cannot produce that; it looks
  like the small operand spilling once C grows. Flagged, not yet modeled.

### §5. Reduction: read-dominated, so it lands on the read-only rate for free

- **Observation:** a reduction reads a full tensor and writes a tiny output. Its effective
  BW sits at the read-only (peak) rate.
- **Why it needs no special term:** with `W ≈ 0`, `min(R,W) ≈ 0`, so the turnaround term
  of §2 vanishes and `T ≈ R/BW_peak` automatically — a reduction is just a read-dominated
  pointwise kernel.
- **Cross-core combine:** when the reduced axis is split across cores there is a ring
  combine; in practice it is provably tiny (bounded by ~`cores × per-elem`, sub-noise),
  so it is carried but effectively inert.

### §6. Transport ops are copies with an access-pattern effective BW

- **Observation:** `transpose`/`cat` move the *same* bytes as a plain copy (they lower to
  a `clone`), yet run at a different effective BW — `transpose` faster, `cat0` slower.
- **Question:** what sets that per-op rate?
- **Hypothesis:** how the copy touches the 64-elem stick (restickify vs sub-stick scatter)
  sets an effective BW, readable from the IR load index.
- **Model:** per-op `BW_eff` override (`restickify` ~116, `stick_scatter` ~60).
- **Figure:** effBW per transport op vs the `neg` copy baseline.
- **Status:** `restickify` settled; `stick_scatter`/`reduce_outer` thin (aspect / reduced-
  dim sweeps pending).

---

## Part III — Matmul: memory *and* compute

### §7. Setup: matmul is not explained by HBM alone

- **Observation:** unlike every op above, matmul time is far larger than its HBM bytes
  predict, and the gap grows with `M·N·K`. There is clearly a compute term.
- **Question:** how do HBM and compute *combine* into one kernel time?
- **Strategy:** we already understand HBM (Parts I–II), so isolate the terms in order —
  **HBM first** (make it dominate), then compute, then how they overlap. Each step
  subtracts only already-validated terms.

### §8. The HBM term — isolated with compute-free matmuls

- **Observation:** in matmuls where compute is negligible (very thin K → output-write-
  dominated; very thin M → operand-read-dominated), the time tracks bytes — but reads and
  writes appear to run at *different* rates.
- **Hypothesis:** `HBM = R/BW_r + W/BW_w + α·min(R,W)` (two-rate + the §2 turnaround).
- **Experiment:** the **rank-2 (R,W) grid** (`run_matmul_gamma_sweep.sh` BW) — shapes
  spanning read-dominated and write-dominated corners so `BW_r`, `BW_w`, `α` separate.
  *(Thin/fat aspect ratios enter here as the isolation tool.)*
- **Figure:** measured vs modeled HBM across the R:W plane.
- **Open:** the earlier fit gave `BW_w = 156 > 150` (physically impossible for a write) —
  the compute-free fit had absorbed compute-overlap; re-fit on the full model (subtract the
  §11 overlap), which this grid + γ enables.

### §9. A split penalty appears — and it is tile-spill, not fanout

- **Observation:** on balanced high-core matmuls the base HBM+compute model leaves a
  residual that **grows with the per-core tile size** (large `M/m`, `N/n`), reaching tens
  of percent.
- **Question:** is the extra cost caused by the number of cores an operand fans out to, or
  by the size of each core's tile?
- **Experiment:** the re-read sweep varies fanout at a fixed small tile (→ ~0 effect,
  **fanout falsified**) and the tile size at fixed fanout (→ the residual).
- **Model:** a per-core operand tile past on-chip capacity is re-streamed from HBM:
  `spill = |A|·f(M/m) + |B|·f(N/n)`, `f` a saturating log with a knee at ~448 rows/cols.
- **Figure:** residual vs per-core tile size, showing the knee.

### §10. The compute term

- **Observation:** with HBM (§8) + spill (§9) subtracted, the remaining time scales as
  `MACs / cores` — i.e. work per core.
- **Hypothesis:** `compute = MACs / cores / peak`.
- **Experiment:** a **compute-dominant cores-scan** (`run_matmul_gamma_sweep.sh` GD, large
  K) — `peak` is the slope of `T` vs `1/cores`, which is *independent of the overlap term*
  (§11), so it escapes the peak/γ identifiability trap.
- **Figure:** `T` vs `1/cores`; slope → `peak` (~1140 MAC/ns/core).

### §11. Compute and HBM overlap — they do not simply add

- **Observation:** `compute + HBM` (added) **over-predicts** at high core counts / balanced
  splits; the real kernel is faster than the sum.
- **Question:** do the memory transfers hide behind the systolic compute?
- **Hypothesis:** `T = compute + HBM − γ·min(compute, HBM)` — the smaller of the two is
  partly overlapped.
- **Experiment:** an **HBM-dominant, spill-free cores-scan** (`run_matmul_gamma_sweep.sh`
  GH, small tiles) — `γ` from the ratio of this scan's compute-coefficient to §10's, with
  spill held out so it can't leak into `γ`.
- **Figure:** measured vs additive (`γ=0`) vs overlap model across cores.
- **Status:** overlap is real; the **value** of `γ` is what this sweep pins (it was a
  non-identifiable ridge before).

### §12. Where the base model breaks: shape-dependent residuals

*(Observed only once §8–§11 give a base model to leave residuals against — inserted here
in logical order, each starting from a residual in the data.)*

- **§12a. Non-power-of-2 N.** *Observation:* the base model **inverts the ranking** of
  `N=6144` vs `8192`, and measured time is non-monotonic in `N`. *Question/experiment:*
  `run_nonpow2_n_sweep.sh` steps `N` so the per-core tile `N/n` crosses a 64-stick edge →
  a **padding sawtooth** the byte count (on stick-aligned full-N) misses. *Model/flag:* the
  per-core-tile stick rounding.
- **§12b. Tiny tiles / extreme forced splits.** *Observation:* very small `M/m` or `N/n`
  and extreme skewed splits carry a fixed-overhead floor and systolic under-fill the base
  model doesn't cover — bounded and flagged as out-of-planner-regime.

---

## Part IV — Coarse tiling *(next report — deferred)*

Placeholder: the fused-kernel reframe (external input counted once), rows-driven underfill
(`rpc`), and the categorical LX-spill. Already largely isolated; written up separately.

---

### Appendix — reproducibility

- **Offline scoring:** `notes/eval_model.py` recomputes accuracy for any model version from
  the stored `(features, measured_time)` dataset — no hardware. `--params k=v` re-scores a
  proposed parameter instantly.
- **Figures:** `notes/plot_report.py` regenerates every figure from `sweep_records.json`.
- **Sweeps:** each section names the sweep script that produced its data.
