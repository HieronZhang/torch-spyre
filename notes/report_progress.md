# Report-writing progress tracker (autonomous loop state)

**Goal:** finish `notes/cost_model_report.md` §6–§12 as full external-reader prose
(no internal jargon), each with figure(s) and an end-of-section data table with
per-point error, matching the observation-first style of §1–§3. Every claim must
survive at least one adversarial agent challenge before it goes in.

**Hard constraints (standing user rules):**

- External audience: define every term; no internal words ("grand sweep", script
  names as jargon, "spill" without definition, dev metrics like "RMS 5→2%").
- Be VERY conservative. Launch an adversarial agent to challenge any mechanism/
  parameter claim; address every challenge or downgrade to hypothesis + the
  deciding experiment.
- No unmodeled residual left silently — flag it explicitly if not modeled.
- Do NOT git commit (user manages git).
- No local hardware: cannot run sweeps. Write handoff scripts; write report from
  EXISTING data in `notes/sweep_records.json` (555 records).

**Section order (dependency chain — do NOT skip):**
§6 transport → §7 matmul setup → §8 HBM → §9 split/tile-spill → §10 compute →
§11 overlap → §12 shape residuals. Part IV (coarse tiling) deferred to next report.

## Status log (append newest at bottom)

- 2026-07-10: §1–§5 written & lint-clean. §6–§12 exist as skeleton bullets only.
  Starting: (a) self-trigger cron, (b) reduction ROWS gap sweep script,
  (c) §1–§5 impl-sync verification, (d) §6 full write.
- 2026-07-10: DONE — cron set (every 5 min, session-only); §1–§5 verified synced
  to cost_model.py (0 mismatches; fixed 2.15e-7→2.148e-7); reduction ROWS gap
  sweep written (run_reduction_rows_sweep.sh). §6 Transport WRITTEN + reviewed by
  adversarial agent BEFORE commit: verified transport→`clone` in IR; transpose=116
  flat (settled number, mechanism open); cat1 default (~7% optimistic); cat0 +
  transpose_outer BOTH flagged size-dependent (equal footing — adversarial caught
  the prior inconsistency). fig6_transport.png added. Handoff sweep for the two
  size-dependent copies: run_transport_shape_sweep.sh. Table = 13 pts, RMS 5.6%.
  NEXT: §7–§12 matmul (shared dataset). User: finish ALL 12, no commits.
- 2026-07-10: DONE — §7–§12 matmul WRITTEN with a strong pre-draft adversarial
  challenge; committed ONLY claims that survived. Key honesty corrections baked in:
  (§8) two-rate BW_w=156 is a fit artifact — verified single-rate 150+γ fits the
  realistic regime BETTER (6.87 vs 7.10%); §8 recommends single-rate. (§9) "fanout
  falsified" downgraded (sweeps confounded, no feats) + small-tile over-prediction
  flagged as co-resident residual. (§10) peak clean. (§11) γ NOT pinned by GH sweep
  (GH prefers γ=0, floor-contaminated) — pinned by balanced aggregate, correlated w/
  memory rate. (§12) regime table: realistic 7.1%, K-split −40%, skewed −23%, tiny,
  non-pow2. Figs fig8–fig11 added. External-reader review pass done: added stick/
  systolic/planner/spill/under-fill glosses, moved script name to appendix, added
  §4+§5+§6 error tables. Status doc updated w/ RESOLVED single-rate finding.
  ALL 12 SECTIONS COMPLETE. Lint clean.

## Data-gap handoff (needs run machine; user resting)

- Reduction large-ROWS residual is backed ONLY by `read` + `sumrow` at ROWS=8192.
  Missing: amax/mean/sumcol/sumall at ROWS=8192, and ALL ops at ROWS=4096/16384.
  New script: `run_reduction_rows_sweep.sh` (written, awaiting run).
- Transport `cat0` + `transpose_outer` are size-dependent copies fit on 2–3 square
  shapes only. New script: `run_transport_shape_sweep.sh` (size × aspect, awaiting run).

## STATUS: ALL 12 SECTIONS DRAFTED (2026-07-10)

§1–§12 all have full external-reader prose + figures + end-of-section error tables.
Added §4/§5/§6 tables too (were missing). Lint clean (ruff + PyMarkdown + mypy pass).
Remaining polish for future ticks (NOT re-drafting):
- If reduction/transport gap sweeps get run, fold data in + tighten §5/§6.
- Consider whether to actually flip the model to single-rate 150 (report §8
  recommends it; needs user OK — do NOT change shipped params unprompted).
- Optional: sync cost_model_presentation.md matmul section to the honest regime
  table (status doc already updated).
- A second external-reader pass once the user is back for final wording tweaks.
If nothing above is actionable, the loop should report "done" and stop.

## Checklist per section (repeat for each)

1. Pull that op-category's rows from sweep_records.json; confirm what data exists.
2. Draft prose observation→question→hypothesis→experiment→model→validation.
3. Generate/refresh the figure(s) via `notes/plot_report.py`.
4. Adversarial agent challenge on the key claim; fix or downgrade.
5. End-of-section table: every data point, pred vs meas, err%. Use eval_model.py.
6. Lint: `python3 -m pymarkdown scan notes/cost_model_report.md` + ruff on py.
7. Update this file's status log.

- 2026-07-10 (loop tick): Verified §1–§12 complete, no skeleton bullets, all 11
  figures present, PyMarkdown clean. Fixed a dangling "see the appendix" reference
  (§6) by listing the two queued handoff sweeps in the appendix. Nothing else is
  autonomously actionable (remaining items need the run machine, a model-flip OK, or
  the user present). REPORT DONE — stopping the recurring 5-min loop to avoid idle
  token spend. Restart with /loop or a new cron when there is fresh work.
