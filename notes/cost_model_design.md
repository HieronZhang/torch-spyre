# Spyre Cost Model — Design, Status, and Plan

A self-contained record so work can resume after any context loss. Companion to
[compiler_pipeline_deep_dive.md](compiler_pipeline_deep_dive.md) (how the compiler/IR works)
and the auto-memory `project-goal-compiler-instrumentation.md`.

Repo: `/home/zhang402/torch-spyre/torch-spyre`. Dev machine has **no Spyre runtime/hardware**;
a separate **run machine** (`/home/haoyang/torch-spyre`, same git) executes. Do all
edits/lint here; hand precise commands to the run machine and paste results back.

---

## 0. TL;DR — where we are

- **Goal:** a high-level, *relative* performance model that predicts Spyre kernel device
  latency from the **"LoopLevel IR — AFTER pre-scheduling passes"** graph, to guide
  higher-level compiler optimization. **Not a cycle-accurate simulator.**
- **Measurement is solved.** We can measure deterministic per-kernel **device** latency via
  `SPYRE_PROFILE_SYNC=1` (uses the merge's new `torch_spyre._C.synchronize()`).
- **First-round model is built and CALIBRATED** (`run_cost_model_plan.sh`, 2026-06-12):
  `T = fixed + hbm_bytes/BW_HBM`, with **`fixed≈20µs, BW_HBM≈111 GB/s`** — predicts
  single-input pointwise (gelu/relu/exp/sigmoid) to **~3%**. **LX traffic is treated as ~free**
  (term dropped; see below).
- **Verified:** arithmetic-free (pointwise memory-bound); HBM BW shared, core-independent
  ≥2 cores; LX ~29× faster than HBM (so ~free). **First round = pointwise only**; reductions
  deferred.
- **Open:** 2-input ops run ~20% over — **attributed** (Rung 3b) to a lower 3-stream BW
  (~80 vs 111 GB/s), `fixed` op-independent; fold in a stream-count-aware BW. Broadcast
  reuse still unmeasured.
- **Lesson logged** (memory `claim-discipline-perf-modeling`): don't make mechanism claims a
  single data point can't support; name the controlled experiment instead.

---

## 1. Goal

Predict **relative** device latency of a Spyre kernel from its after-pre-scheduling LoopLevel
IR, so we can rank compiler choices (LX placement, work division, fusion, tiling) without
running. Accuracy target: good enough *ordering*, not absolute ns. A simulator would be
useless overkill.

The IR input per op gives: `ranges` (output iter space), `reduction_ranges`,
`op_it_space_splits` (cores per dim), `device_size`/`stride_map` (sticks), `allocation`
(LX or not — only LX is annotated at this stage; everything else is HBM), and the `inner_fn`
(loads → which tensors, broadcast pattern; the arithmetic). See §6.2 of the deep-dive for how
to read an op.

---

## 2. Design principles (agreed with user)

1. **Relative, not simulator.** Few parameters, coarse. Predict which is faster.
2. **Bandwidth + one fixed term, not per-access latency.** A static-dataflow engine streams
   sticks back-to-back; per-access latency is hidden and shows up once as a **fixed
   per-kernel cost** (≈20 µs measured; pipeline fill+drain + device setup + ~7 µs host
   residue). So model traffic ÷ effective bandwidth, plus that fixed term.
3. **Effective bandwidth folds in the messy parts.** Real per-core BW depends on DRAM
   controller buffering, concurrent cores, row-buffer hits — but Spyre access is stick-aligned
   (128 B), streaming, statically scheduled, so those are *constant* for a given shape and
   collapse into one **effective aggregate BW**. For *relative* predictions with the same
   streaming pattern, the absolute BW **cancels** → robust without modeling the controller.
4. **Verification-driven.** Build the model AND a benchmark that checks each assumption; add a
   term only where a benchmark shows the simple model *misranks*. (See §9.)
5. **One knob at a time, simplest first.** Pointwise before reductions; single op before fused.

---

## 3. The model

Per kernel/bundle (one fixed term):

```
T = fixed + hbm_bytes / BW_HBM          (LX-resident traffic treated as ~free)
```

- `fixed` (≈**20 µs**) — the **lumped fixed per-kernel overhead**: it is *not* purely
  "pipeline fill." It bundles (a) device pipeline fill+drain, (b) per-kernel device setup
  (program load, DMA/core config), and (c) **~7 µs host residue** in our measurement (async
  dispatch + sync-return latency that `SPYRE_PROFILE_SYNC` brackets). **One per kernel**, not
  per op. **It MUST be op-independent** — relu/gelu/exp/sigmoid (all 2-stream) measured the
  same ⇒ same `fixed`. If a fit ever needs a different `fixed` per op, the **model is wrong**
  (mis-counted traffic / a missing structural term), not "fixed is op-specific." Any
  refinement must be a *principled* function of structure (e.g. stream count — same for any
  3-stream op), never an arbitrary per-op fudge.
- `BW_HBM` — effective **aggregate** HBM bandwidth in GB/s (numerically == bytes/ns,
  since 1 GB/s = 1 byte/ns). "Aggregate" = the shared-HBM assumption; the SENCORES sweep
  (§8 rung 5) verifies core count doesn't change it.
- **LX traffic is treated as ~free (no `BW_LX` term).** Rung 4 showed the per-op LX cost
  (~1 µs) sits *below* the run-to-run measurement noise (~5 µs), so a precise `BW_LX` can't be
  resolved; LX-resident tensors barely affect latency (LX is ~29× HBM). `lx_bytes` is still
  computed for inspection but contributes 0 to `T`. (Revisit only if an LX-heavy case
  misranks — would need a larger-tile LX sweep to lift the signal above the noise.)
- `hbm_bytes` / `lx_bytes` — sum over every tensor-arg (each input AND the output) of its
  bytes, attributed to HBM or LX by **allocation propagation** (an op's input memory = its
  producer's output allocation). **LX-placed tensors don't count toward HBM.**
- **Broadcast** inputs are flagged; whether they re-fetch (full traffic) or cache (reduced) is
  a *measured unknown* (§8 rung "broadcast"). Default = conservative full count.

**Per-op vs fused:** model a single op as its own kernel; a fused bundle = sum of its ops'
traffic with **one** fill, and intermediates that stay in LX don't hit HBM. (Our examples run
SDSC-bundle fusion but **NOT** tile fusion — tile fusion needs `spyre_hint` coarse-tile groups,
absent in bare softmax/gelu. Tile fusion is a *separate, later* regime where loop-internal
intermediates become per-tile on-chip scratch.)

**Why two "fusions" — do not conflate:**
- **SDSC bundle fusion** (`spyre_fuse_nodes`): groups ops into one kernel (`sdsc_fused_*`);
  intermediates still live in memory (HBM/LX). ACTIVE in our runs.
- **Tile fusion** (coarse tiling): inner-loop fusion making intermediates per-tile on-chip;
  needs hints. NOT active in our runs.

**Calibrated values (fp16, fitted from `run_cost_model_plan.sh`):** `fixed≈20µs`,
`BW_HBM≈111 GB/s` (rung 1 slope; 2-stream r/w, ≥2 cores, shared & saturated). **LX traffic
is treated as ~free** (no `BW_LX` term — rung 4 signal was below noise). Single-input
pointwise predicted to ~3% across a 16× size range. See §5 for the data.

---

## 4. Measurement methodology (the instrument)

Static dataflow ⇒ device latency is **deterministic**. Strategy: take the **min over N runs**
(host jitter only *adds* time, so min strips it); confirm via collapsed spread.

**Key facts established:**
- **Host floor ~70 µs/call.** Every compiled-fn call pays ~70 µs host overhead (Dynamo guard,
  wrapper, output alloc, ~7 µs async dispatch). Small-op device compute hides under it →
  end-to-end wall-clock CANNOT measure it.
- **Device sync now exists** (merge `#918`): `torch_spyre._C.synchronize(device=None)` blocks
  on the Flex runtime stream. (Pre-merge the c10 sync hooks were no-op stubs; the merge added
  the `SpyreStream`/stream-pool plumbing that holds a `flex::RuntimeStream*` handle to wait on.
  `elapsedTime`/event-timing is still stubbed — we time the host clock *around* a sync.)
- **`SPYRE_PROFILE_SYNC=1`** makes the per-kernel timer block on the device after each launch →
  the registry's per-kernel **min** is real device latency, for ANY op (small included).
- **Don't use end-to-end `net` or the identity baseline for small ops** — host-wrapper
  differences dominate (gave nonsensical negative nets). Read the per-kernel device min directly.

**Env vars (instrument):**
| var | effect |
|---|---|
| `SPYRE_PROFILE=1` | record per-kernel host launch times (registry, atexit table) |
| `SPYRE_PROFILE_SYNC=1` | sync device after each launch → per-kernel **device** latency |
| `SPYRE_PROFILE_FILE=path` | append profile report to file instead of stderr |
| `SPYRE_DUMP_IR=1` | dump ATen FX + LoopLevel IR (before/after pre-scheduling) |
| `SPYRE_DUMP_COST=1` | dump cost-model features + prediction at compile time |
| `LX_PLANNING=1/0` | LX scratchpad planning (now defaults ON) |
| `SENCORES=N` | number of cores (1–32) |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` | force recompile (else cache hit skips dumps) |

---

## 5. Empirical data gathered (calibration set)

All device-side, deterministic (min-of-N), fp16:

| op | size | elements | device min | notes |
|---|---|---|---|---|
| softmax (bundle) | 512×1024 | 0.5 M | 65–71 µs | 5 ops fused |
| softmax | 512×4096 | 2 M | ~200 µs | |
| softmax | 4096×4096 | 16 M | 1594 µs | 8× elems ⇒ 8× time (linear) |
| add (`a+0`) | 512×1024 | 0.5 M | 32.9 µs | 1-in/1-out memory pass |
| add | 512×4096 | 2 M | ~87 µs | |
| add | 4096×4096 | 16 M | 594 µs | |

**Fits** (`latency = fixed + slope × M-elem`):
- **add ≈ 14.8 µs + 36.2 µs/M-elem** (near-perfect)
- **softmax ≈ 16.3 µs + 98.6 µs/M-elem** (slope ≈ 2.7× add ⇒ ~2.7 DDR passes)
- **fixed ≈ 15 µs** shared by both ⇒ op-independent fixed cost. (3-point fit; the 5-point
  gelu ladder in §5.1 refines this to **~20 µs** — use that.)
- One 0.5 M-elem (2 MB) HBM round-trip ≈ **18 µs** ⇒ **~110 GB/s** effective.

### 5.1 gelu op-ladder calibration (`run_cost_model_plan.sh`, 2026-06-12, fp16)

The pointwise ladder, all DEVICE min via `SPYRE_PROFILE_SYNC`:
- **Rung 1 (gelu size sweep, ROWS=512):** 0.26M→29.5µs, 0.52M→39.2, 1.05M→59.0, 2.1M→93.0,
  4.19M→171.5. Clean linear ⇒ **`fixed≈20.1µs`, `BW_HBM≈111 GB/s`** (slope 35.9µs/Melem;
  2 passes × 2 B). Predicts to ~3%.
- **Rung 2 (arithmetic):** relu 38.4, gelu 38.2, exp 38.4, sigmoid 34.5 (≈equal) ⇒
  **arithmetic FREE, memory-bound. CONFIRMED.** (Also confirms `fixed` is op-independent.)
- **Rung 3 (traffic):** gelu(1-in) 41.0, mul(2-in) 58.1, add 57.7. +1 input ≈ +17µs.
  **BUT** measured mul is ~20% OVER the byte-linear model — UNATTRIBUTED (could be `BW` or a
  per-stream fixed cost; a single size point can't tell). → Rung 3b (below).
- **Rung 4 (LX chain, all-LX):** gelu depth 1/2/4/8/16 → 39.6/34.4/34.7/39.3/43.0 — nearly
  FLAT and **non-monotonic** (16 chained gelus ≈ 1). The per-op LX cost (~1 µs) is *below* the
  run-to-run noise (~5 µs), so `BW_LX` can't be resolved (the ~3200 GB/s "fit" is an artifact).
  Decision: **treat LX as ~free** (drop the term); qualitatively LX is ~29× HBM.
- **Rung 5 (SENCORES):** 1→60.3, 2→40.5, 4→43.5, 8→41.9, 16→40.8, 32→40.5. 1→2 helps then
  FLAT ⇒ **HBM BW SHARED, saturates ~2 cores ⇒ core count NOT a direct model term.**

**Rung 3b (queued)** — controlled multi-input attribution, holding `fixed` op-independent:
a `mul` size sweep (fit its own `fixed`,`BW`), plus EQUAL-total-bytes/different-stream pairs
(`mul[512×1024]` 3-stream == `gelu[512×1536]` 2-stream = 3,145,728 B; and the 6 MB pair
`mul[512×2048]`==`gelu[512×3072]`). The model predicts each pair EQUAL; any gap is the pure
stream-count effect. Gap constant 3MB→6MB ⇒ fixed per-stream cost; gap doubles ⇒ per-byte BW.

**Core division — effect (Rung 5 + reasoning):** for memory-bound pointwise, core *count*
(≥2) has **no direct** effect (BW-bound, flat). It matters **indirectly** via: (1) **LX fit** —
per-core tile = `total/cores`; placement only succeeds if it fits ~1.6 MB, flipping
`lx↔hbm` bytes (the cliff; already in the model through `allocation`); (2) load balance
(uneven splits → max-core); (3) **reductions** — splitting the reduced axis adds a cross-core
combine (a term still owed). Direct access-pattern/locality effects are UNVERIFIED.

**LX experiment** (softmax[512×1024], device min): **LX on = 71.05 µs, off = 91.23 µs ⇒ −20 µs
(22%)**. LX keeps the `sub` intermediate in SRAM, removing its ~18 µs HBM round-trip (matches
the add round-trip cost). Implication: LX planning subtracts a tensor's HBM passes; fusion and
LX are complementary (fusion = one kernel, LX = keeps intermediate off DDR). Predicted cliff:
benefit vanishes once per-core `sub` tile (`total/cores × 2 B`) outgrows ~1.6 MB LX (~8192²).

---

## 6. What's built (files)

**Instrument (committed earlier):**
- `torch_spyre/execution/profiling.py` — `kernel_timer` ctx-mgr (wraps `SpyreSDSCKernelRunner.run`
  in `kernel_runner.py`); `SPYRE_PROFILE` / `SPYRE_PROFILE_SYNC`; `format_report`;
  `set_report_at_exit`; `_device_synchronize` → `_C.synchronize()`.
- `torch_spyre/execution/bench.py` — `measure_device(fn, runs, warmup)` (device-side, default,
  reads registry min, warmup discarded); `measure_latency` (host e2e, kept for user-latency);
  `device_sync`, `LatencyStats`, `net_latency_us`.
- `torch_spyre/_inductor/dump_common.py`, `dump_fx_graph.py`, `dump_loop_ir.py` — IR dumps
  (`SPYRE_DUMP_IR`), wired into `passes.py` (`CustomPostPasses` for FX; `CustomPreSchedulingPasses`
  before/after for LoopLevel).

**Cost model (this round):**
- `torch_spyre/_inductor/cost_model.py` — PURE model: `OpFeatures`, `ArgTraffic`, `CostParams`
  (**calibrated `fill_ns=20000, bw_hbm_gbps=111`; LX treated as free, no `bw_lx`**),
  `predict_ops`, `predict_op`, `explain`. No torch deps ⇒ path-loadable/testable.
- `torch_spyre/_inductor/dump_cost_model.py` — `extract_features(operations)` over live IR
  (cores from `op_it_space_splits`; per-arg bytes + LX/HBM via allocation propagation; broadcast
  flag). Hook `dump_cost_model` wired after the AFTER LoopLevel dump in `passes.py`;
  `SPYRE_DUMP_COST=1` prints features + prediction.
- `examples/bench_ops.py` — device-side pointwise ladder. `BENCH_OP=gelu|relu|sigmoid|exp|mul|add`,
  `BENCH_DEPTH=N` (unary chain for LX-BW sweep), `BENCH_LX_ALL=1`
  (`config.allow_all_ops_in_lx_planning=True` → all ops LX-eligible, `allocator.py:97`),
  `BENCH_ROWS/COLS/RUNS/WARMUP`.
- `examples/bench_softmax.py` — device-side softmax bench (prints `LX_PLANNING`).
- `examples/bench_sweep.py` — size sweep (currently host-e2e; TODO align to device-side).

Local `uv` env at `.venv` (Python 3.12, torch 2.11 cpu, ruff, mypy) for lint + standalone tests.
Full `pip install -e .` not possible here (no SDK); on the run machine, rebuild C++ after a
merge with `python setup.py build_ext --inplace` (see `_C` rebuild note — missing symbols like
`ElementArrangement` mean a stale `.so`).

---

## 7. Microarchitecture facts & assumptions

Known / assumed (verify in §9):
- 32 cores (SENCORES). Each core has **2 MB LX scratchpad** (~1.6 MB usable).
- **Stick = 128 B = 64 fp16 elems**; within-stick is the last device dim; work division won't
  split inside a stick.
- Two execution units: **`pt`** (matmul/PE array), **`sfp`** (vector/SIMD — pointwise &
  reductions). Pointwise is memory-bound ⇒ unit throughput not yet modeled.
- Memory hierarchy: HBM (shared) ↔ per-core LX ↔ compute. LX-placed tensor's per-core tile must
  fit ~1.6 MB.
- Work division (`op_it_space_splits`) distributes the iteration space across cores; cores run
  in parallel ⇒ latency ≈ per-core time (balanced).
- No public microarch spec in-repo ⇒ **infer parameters from micro-benchmarks** (§8/§9).

---

## 8. The op ladder (one knob per rung) + run commands

Run on the run machine. Pair with `SPYRE_DUMP_COST=1` to print predictions.

```bash
# 1) fit fill + BW_HBM (single op, size sweep). slope→BW_HBM, intercept→fill
for n in 512 1024 2048 4096; do BENCH_OP=gelu BENCH_COLS=$n python examples/bench_ops.py; done
# 2) arithmetic-free? relu vs gelu same size (equal ⇒ memory-bound; gelu slower ⇒ add compute term)
BENCH_OP=relu python examples/bench_ops.py ; BENCH_OP=gelu python examples/bench_ops.py
# 3) traffic counting: 1-input vs 2-input (mul ~1.5× gelu: 3 passes vs 2)
BENCH_OP=mul python examples/bench_ops.py
# 4) BW_LX: unary chain depth, all intermediates in LX. slope vs N = 2·|x_tile|/BW_LX
for d in 1 2 4 8; do BENCH_LX_ALL=1 BENCH_DEPTH=$d python examples/bench_ops.py; done
#    (first VERIFY via SPYRE_DUMP_IR=1 that intermediates got `lx` and N ops survived)
# 5) shared vs per-core HBM BW (decides whether `cores` enters the model)
for c in 1 2 4 8 16 32; do SENCORES=$c BENCH_OP=gelu python examples/bench_ops.py; done
# broadcast reuse: compare an op with a [1,N] broadcast input vs a full [M,N] input
```

---

## 9. Verification checklist (earn the right to stay simple)

- [x] effective `BW_HBM` constant across shapes (size-sweep linearity) — rung 1 ✓ (~111 GB/s)
- [x] `fixed` is a real op-independent intercept — rung 1 ✓ (~20 µs; rung 2 confirms op-indep)
- [x] BW shared vs per-core — rung 5 ✓ (SHARED; flat ≥2 cores; cores not a direct term)
- [~] traffic = Σ inputs + output — rung 3: holds for 1-in; **2-in ~20% over, UNATTRIBUTED** → rung 3b
- [x] arithmetic free for pointwise (relu == gelu?) — rung 2 ✓
- [ ] **broadcast** reuse: cached vs re-fetched — broadcast rung (not yet run)
- [x] **LX cost** from chain-depth — rung 4 → per-op LX cost below noise ⇒ **LX treated as ~free** (term dropped)
- [x] cost-model `extract_features` matches the IR — confirmed (`SPYRE_DUMP_COST` op counts/bytes)
- [ ] **multi-input fill-vs-BW** (equal-bytes, 2 vs 3 streams, two budgets) — **rung 3b (queued)**

---

## 10. Plan / next steps

1. ~~Rungs 1–5 + 3b~~ DONE — `fixed≈20µs`, `BW_HBM≈111` (2-stream); arithmetic-free, shared-BW
   verified; LX treated as ~free; multi-input attributed to 3-stream BW ~80. `CostParams` updated.
2. **Run Rung 3b** (`run_cost_model_plan.sh` includes it) → attribute the 2-input ~20% gap to
   a principled per-stream term (fixed-side or BW-side), keeping `fixed` op-independent.
3. Resolve **broadcast** (cached vs re-fetched) — needs a `[1,N]`-input bench (not yet built).
4. Validate the full pointwise model on `mul`/`add`/`gelu` across sizes vs `SPYRE_DUMP_COST`.
5. **Then** extend to reductions: read-dominated traffic + the **cross-core combine** term
   (when the reduced axis is split — tied to core division). Re-validate on `mean`, then softmax.
6. Later: tile-fusion regime (hinted examples), matmul (`pt` unit / compute-bound), the LX
   capacity cliff as a hard constraint.

---

## 11. Open questions

RESOLVED: shared-vs-per-core HBM BW (rung 5 → SHARED, cores not a direct term); LX is **~free**
(rung 4 → per-op LX cost below noise; **term dropped**, qualitatively ~29× HBM); arithmetic-free
(rung 2); multi-input ~20% gap **attributed** (Rung 3b → lower 3-stream BW ~80, `fixed` op-indep).

Still open:
- Fold the **stream-count-aware `BW_HBM`** (2-stream ~111, 3-stream ~80) into the model/code.
- **Broadcast** reuse factor (cached vs re-fetched) — needs a `[1,N]`-input bench.
- **Reduction combine** cost (deferred) — scales with reduction-axis split (core division).
- LX precise BW unresolvable here (signal < noise) — only revisit with a larger-tile LX sweep
  if an LX-heavy case ever misranks (low pri).
- **Why is effective `BW_HBM` (~111) only ~half the >200 GB/s DRAM peak?** **GUESS (unverified):**
  Rung 5 was flat for ≥2 cores ⇒ ~2 cores already saturate the rate, so the limiter looks like
  a **shared resource *upstream* of the DRAM** (on-chip interconnect / HBM-controller path that
  feeds the cores) — the >200 GB/s DRAM isn't the wall, the path to it is. (Plus R+W turnaround /
  DRAM efficiency; 2→3 streams dropping 111→80 hints at concurrent-stream contention.) To test:
  read-only (`sum`) vs write-only (`fill`/`zeros`) vs R+W (`gelu`); and 1-core BW × N vs the cap.
- Does effective `BW_HBM` stay constant across access patterns / dtypes for ranking? (verify).
- LX capacity: exact usable size per core and how work division maps tiles into it (the cliff).
- `fixed`'s ~7 µs host residue: is it stable across kernels / would a tighter device-only timer
  (event timing, still stubbed) change it?
