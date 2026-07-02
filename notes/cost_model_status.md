# Spyre Cost Model — Current Status (2026-07-01)

Living status/handoff doc — the single up-to-date source. Read this first on resume.
Companions: [cost_model_summary.md](cost_model_summary.md) (model + matmul plan),
[cost_model_design.md](cost_model_design.md) (older full design; partly stale).

Goal: a HIGH-LEVEL **relative** cost model over the after-pre-scheduling LoopLevel IR,
to guide optimization decisions (LX placement, whether/how to coarse-tile). Accurate
*ranking* + ±15-20% is the bar; re-sweepable when the compiler/HW changes. No local HW —
edits done locally, commands handed to a run machine, logs pasted back.

## Golden measurement (IMPORTANT)

Use the **AIU profiler**: `torch.profiler` ProfilerActivity.PrivateUse1, "Self SPYRE"
per-kernel device time. The harness ([profile_ops.py]) does this. The new runtime image
leaves the kernel event **name BLANK**, so we now classify the kernel **by exclusion**
(device time that is NOT Memset and NOT Memcpy) — a name match (`sdsc_fused`) silently
returned 0. Do NOT use `SPYRE_PROFILE` (host launch) or `SPYRE_PROFILE_SYNC` (needs
`SPYRE_PROFILE=1` too, and is still a host wall-clock) — they are not the golden path.

## The model (torch_spyre/_inductor/cost_model.py)

```
T = fill + [ hbm ] / eff_underfill + matmul_compute + combine + c_loop*L
hbm      = (R+W)/BW_PEAK + a*min(R,W)             (pointwise/reduction/transport)
         = R/mm_bw_read + W/mm_bw_write + a*min(R,W)   (matmul, TWO-RATE)
matmul_compute = MACs/cores/(mac_peak * pt_eff)   (ADDITIVE)
combine  = (k-1)*out_elems*psum_per_elem          (reduction ring / matmul PSUM)
```

Params + validation status:
| param | value | status |
|---|---|---|
| `bw_peak_gbps` | 150 | ✅ pointwise/reduction (~2-7%) |
| `rw_turnaround_ns_per_byte` (a) | 0.00574 | ✅ pointwise balanced |
| `mm_bw_read_gbps` / `mm_bw_write_gbps` | 143 / 156 | ✅ NEW, fit on M1 (±4%, cores=32) |
| `mac_peak_per_core_ns` | 1536 | ⚠️ datasheet; matches at cores=32, off at cores=16 |
| `underfill_pass_rows` / `_target_passes_pointwise` / `_exponent` | 8 / 2 (r_full=16) / 0.35 | ✅ chain sweep |
| `underfill_target_passes_matmul` | 8 (r_full=64) | ⚠️ pt_eff=1 for all tested (M/m>=128); NOT the split lever -- split penalty is FANOUT, see below |
| `c_loop_ns` (reduction-dim tiling) | 860 | ✅ reduction K-sweep |
| `c_loop_pointwise_ns` | 0 | ✅ Section-A chain flat |
| `psum_per_elem_ns` | 0.14 | ❌ matmul PSUM 3-10x over (see below) |

## Findings by op category

**Pointwise / reduction** — turnaround HBM model, ~2-7%. Settled.

**Coarse-tiling (chain / softmax / matmul row-tiling)** — output-dim tiling fuses ops into
one loop with intermediates LX-resident. Two mechanisms: `c_loop*L` (per-iteration, ~0 for
pointwise, 860ns for reduction-dim) and `eff_underfill` (per-tile-SIZE derate, p=0.35,
saturates rc>=16). `rc=1` "anomaly" = planner switches row-split -> col-split, so
`tile_rows_per_core` now uses the ROW-dim split (d0), not total cores. **Now measured on the
golden profiler** (2026-07-01): `softmax_row_tiling` [16384x4096] +11.8% (model over),
`matmul_row_tiling` [2048^3] -9.9% (model under) — both within bar, opposite signs, so the
fused-LX-resident structure is validated (arg0 confirmed re-read twice in softmax: read-once
would be -17%, twice +12%). Two small residuals for the backlog, NOT knobs: (a) the turnaround
term is too big in a fused reduce-softmax (reads and the single write are temporally separated
-> ~L direction-switches, not min(R,W)-proportional); (b) coarse-tiled matmul wants a small
per-tile overhead (~10us/tile), distinct from the reduction-dim c_loop=860.

**Matmul** — `T = compute + hbm + psum`, additive (K-sweep at cores=32).
- HBM: **fixed to two-rate 143/156** (reads slower than writes), ±4% on the compute-free
  M1 sweep. Read-dom went -8% -> ~0%.
- Compute: datasheet peak matches at **cores=32 for BALANCED splits**. Split sweep
  (run_split_sweep.sh, k=1, 3 sizes) -> additive model accurate a few % (SP3 2048^3:
  -2.8/+1.7/+2.1%) to ~15% (SP2 large M) for m,n in {2,4,8} -- the region a planner picks.
  BUT the split alone swings true time up to **1.9x at fixed compute/hbm/bytes** (V-shape,
  min 4x8). Penalty onset is a **FANOUT threshold, NOT rows/core**: B-fanout m>8 (m=16: 1.65
  SP1 / 1.85 SP2, replicated) and A-fanout n>16 (n=32: 1.91, SP1). Decoupled from tile height
  by m8-vs-m16 at equal M/m=256 (0.98 vs 1.65 -> tracks m; pt_eff=1 for all, so the existing
  underfill is blind to it). **`(fanout/8)^0.75` FALSIFIED**: predicts 1.68 at n=16 where
  actual is 1.12, and saturates (m16=m32=1.65) instead of growing. CONFOUND remaining: n=32
  also = N/n=64 (1 stick), so fanout vs per-core-width not yet separated. NO param change
  until run_decouple_sweep.sh (vary MATRIX dim at FIXED 4x8) isolates them.
- PSUM: **model 3-10x over** (k=8 total +472%). But CANNOT be isolated until hbm+compute
  are accurate (varying k at fixed cores changes (m,n) -> moves hbm+compute). psum is LAST.

**Transport (restickify)** — the 4th category. A transpose/cat lowers to a Pointwise copy
(model counts R=W=data) but codegens to RESTICKIFY_OP. Data: `transpose` bw ~116 (11%
FASTER than a `neg` copy at 105 — restickify pays less turnaround) and SHAPE-independent;
`cat0` slow (bw 63) because the concat dim lands next to the stick (`[2048,32,2,64]`, fine
interleave) vs `cat1` (`[2,32,2048,64]`, outer, bw 108). Fix (deferred): a per-op eff-BW
mapping (transport ~116); needs restickify detection in the extractor + a fuller sweep.

## Methodological lessons (do NOT repeat)

1. **Never `measured - model_term` to isolate another term** — circular; broke twice (psum,
   then the CM compute isolation gave pt_eff>1 nonsense).
2. **Measure at matching core counts** — the CM sweep used cores=16 (for re-read-free) but
   hbm/additivity were calibrated at cores=32; incomparable. Re-read-free (fanout<8) and
   cores=32 are mutually exclusive at k=1, so the split must be attacked WITH re-read present.
3. **Trust measured data, not the in-tree work_division.py cost model** — it was rewritten in
   the merge (pt_eff target 8->5 exp 0.25, cohort `(fanout/8)^0.75`, per-core psum). It's a
   relative *ranker*, not an absolute-time predictor; use its FORMS as candidates, calibrate
   constants to our sweeps.

## Tooling built

- **Harness** `docs/source/user_guide/examples/profile_ops.py` (BENCH_OP=...). Ops: pointwise
  (neg/gelu/mul/add3/add4/...), reductions (sumrow/sumcol/amax/...), broadcasts, `chain`,
  `softmax_row_tiling`, `matmul_row_tiling` (coarse tiling), `mm` (plain), `mmwd` (FORCED
  split via WD_M/N/K), `transpose`/`cat0`/`cat1`/`transpose_outer` (transport), coarse-tiled
  reductions. Knobs: BENCH_ROWS/COLS/N, BENCH_TILES, WD_M/N/K, SENCORES, LX_PLANNING.
- **Sweep scripts** (same dir): `run_matmul_validate_sweep.sh` (M1 hbm / M2 compute / M3 psum
  - E2E smoke), `run_transport_sweep.sh` (T1/T2), `run_compute_isolate_sweep.sh` (CM1/CM2 —
  the flawed cores=16 one), `run_split_sweep.sh` (SP1/2/3 — cores=32, QUEUED), `run_tiling_
  terms_sweep.sh` (A/B chain + superseded C-H), `run_all_sweeps.sh` (chains several).
- **Parser** `notes/parse_sweep_logs.py` -> `notes/sweep_records.{json,csv}`. Merges by
  `log:lineno` (idempotent). All runs to date are recorded there.

## Profiling database (run_db_sweep.sh)

One command rebuilds the whole measurement DB so any model change is validated by diffing
model vs recorded `kernel_us` (never re-derived): `bash docs/source/user_guide/examples/
run_db_sweep.sh` (~170 runs, ~30-45 min; runs are now CPU-time-bound). It chains every sweep in
the matmul isolation order + global op coverage, then auto-runs the parser. New DB scripts:
- `run_hbm_ops_sweep.sh` (HB1/2/3) — per-op-type effective BW: pointwise arity, reduction axis,
  broadcast one-time-load. (transport BW = `run_transport_sweep.sh`.)
- `run_matmul_compute_sweep.sh` (MC1/2/3) — **step 2**: mac_peak at LOW cores (compute-dominant,
  HBM byte-subtracted), cores-scaling (per-core-peak / cores=16 anomaly), peak across shapes.
  KEY: compute dominates only at low cores, NOT big K (K scales compute AND reads equally).
- `run_matmul_psum_sweep.sh` (PS1/2) — **step 4**: k-sweep at FIXED (m,n) (no re-read confound) +
  out-size dependence. Run LAST (needs compute+split pinned).
- `run_coarse_tiling_sweep.sh` (CT1/2/3) — softmax/matmul row-tiling x tile count + chain/ctsum LX.
- `run_db_sweep.sh` — master; `unset SECTIONS` so every child runs full; matmul_validate gets
  `SECTIONS=M1` (HBM only). Split (step 3) reuses `run_split_sweep.sh` + `run_decouple_sweep.sh`.

## Merge / runtime notes

- New image: kernel-name-blank -> by-exclusion classifier (fixed in profile_ops + profile_test).
- Post-merge codegen changed (superdsc/unroll/fusion/scratchpad): **planner-split `mm` data is
  stale**; forced `mmwd` data is fine. Pre-merge `sweep_records` times may have shifted.
- E2E coarse-tile examples on the new image: `softmax_row_tiling` ✅, `matmul_row_tiling` ✅,
  **`matmul_k_tiled` FAILED** (`dxp_standalone --bundle ... SIGABRT` — toolchain, not our code;
  report to manager).

## Immediate next steps (in order)

1. **Decoupling sweep** — split sweep DONE (fanout penalty found at m>8 / n>16, up to 1.9x;
   `(fanout/8)^0.75` falsified; confounded with per-core tile width). Next: `run_decouple_
   sweep.sh` varies the MATRIX dim at FIXED 4x8 split (DC1 sweep N -> N/n 64..512; DC2 sweep
   M -> M/m 128..2048) to separate per-core-tile underutilization from fanout, THEN commit a
   split-penalty derate. NO param change until decoupled.
2. **Coarse-tiling golden measurements** — `softmax_row_tiling` (+11.8%) and
   `matmul_row_tiling` (-9.9%) DONE, both within bar. `chain` still pending.
3. **THEN psum** (only after 1+2 pin compute).
4. Fold into `cost_model_design.md` + `notes/README.md` (canonical write-up) — still pending.
5. Transport per-op eff-BW mapping (needs restickify detection + fuller sweep).
