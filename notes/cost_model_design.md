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
- **First-round model is built and matches data to ~4%:**
  `T = fill + hbm_bytes/BW_HBM + lx_bytes/BW_LX`.
- **First round = pointwise ops only.** Reductions (cross-core combine) deferred.
- **Next:** run the op-ladder benches to fit `fill, BW_HBM, BW_LX` and verify assumptions.

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
2. **Bandwidth + one fixed fill, not per-access latency.** A static-dataflow engine streams
   sticks back-to-back; per-access latency is hidden and shows up once as **pipeline fill**
   (the measured ~15 µs fixed term). So model traffic ÷ effective bandwidth, plus fill.
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

Per kernel/bundle (one pipeline fill):

```
T = fill + hbm_bytes / BW_HBM + lx_bytes / BW_LX
```

- `fill` — one-time pipeline-fill latency (≈15 µs measured). **One per kernel**, not per op.
- `BW_HBM`, `BW_LX` — effective **aggregate** bandwidths in GB/s (numerically == bytes/ns,
  since 1 GB/s = 1 byte/ns). "Aggregate" = the shared-HBM assumption; the SENCORES sweep
  (§8 rung 5) verifies whether core count changes it.
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

Validation so far: gelu[512×1024] (1-in/1-out, all HBM, 2 MB traffic) predicts **34.07 µs**
with `fill=15µs, BW_HBM=110 GB/s` vs measured `add`≈**32.9 µs** (4%).

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
- **fixed ≈ 15 µs** shared by both ⇒ op-independent pipeline-fill/launch.
- One 0.5 M-elem (2 MB) HBM round-trip ≈ **18 µs** ⇒ **~110 GB/s** effective.

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
  (`fill_ns=15000, bw_hbm_gbps=110, bw_lx_gbps=1000` — placeholders), `predict_ops`,
  `predict_op`, `explain`. No torch deps ⇒ path-loadable/testable.
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

- [ ] effective `BW_HBM` constant across shapes (size-sweep linearity) — rung 1
- [ ] `fill` is a real fixed intercept — rung 1
- [ ] BW shared vs per-core (does latency ∝ 1/cores?) — rung 5
- [ ] traffic = Σ inputs + output — rung 3
- [ ] arithmetic free for pointwise (relu == gelu?) — rung 2
- [ ] **broadcast** reuse: cached vs re-fetched — broadcast rung
- [ ] **BW_LX** from chain-depth slope; intermediates actually placed in LX — rung 4
- [ ] cost-model `extract_features` matches the IR (eyeball `SPYRE_DUMP_COST` vs the dump)

---

## 10. Plan / next steps

1. **Run rung 1**, fit `fill` + `BW_HBM` (optionally build a least-squares fitter that ingests
   the sweep numbers). Update `CostParams` defaults.
2. Run rungs 2–3 (arithmetic-free, traffic counting) → confirm the pointwise traffic model.
3. Run rung 4 (+`SPYRE_DUMP_IR` check) → fit `BW_LX`; rung 5 → settle shared-vs-per-core.
4. Resolve **broadcast** (cached vs re-fetched) and bake the factor in.
5. Validate the full pointwise model on `mul`/`add`/`gelu` across sizes; check `SPYRE_DUMP_COST`
   predictions vs measured per-kernel device min.
6. **Then** extend to reductions: add the read-dominated traffic + the **cross-core combine**
   term (when the reduced axis is split). Re-validate on `mean`, then softmax (whole bundle).
7. Later: tile-fusion regime (hinted examples), matmul (`pt` unit / compute-bound), the LX
   capacity cliff as a hard constraint.

---

## 11. Open questions

- Shared vs per-core HBM BW (rung 5) — biggest single unknown for the model's `cores` handling.
- Broadcast reuse factor.
- `BW_LX` absolute value (rung 4) and whether LX is ~free vs HBM (slope ≈ 0?).
- Reduction combine cost model (deferred).
- Does effective `BW_HBM` stay constant enough across access patterns for relative ranking, or
  do we need a per-op correction? (verify, don't assume).
- LX capacity: exact usable size per core and how work division maps tiles into it.
