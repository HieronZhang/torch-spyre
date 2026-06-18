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
- **Model RE-ANCHORED on the golden profiler kernel time** (`run_profile_sweep.sh`, §5.3):
  `T_kernel = hbm_bytes/BW_HBM` with **`fill≈0, BW_HBM≈102 GB/s`** (balanced 1R+1W; R²≈1.0).
  The old `fixed≈20µs` was non-deterministic Memset/host **overhead** (size-scaling), NOT kernel
  cost — now excluded. **LX traffic ~free.** read/write split + reductions pending sweep §B–F.
- **Verified:** arithmetic-free (pointwise memory-bound); HBM BW shared, core-independent
  ≥2 cores; LX ~29× HBM (so ~free); **broadcast AND scalar inputs are cached → ~free**
  (Rung 6, encoded). **First round = pointwise only**; reductions deferred.
- **Bandwidth — the ~111 cap EXPLAINED (Rung 8):** read-only ~**176** GB/s (86% of the 204.8
  LPDDR5 peak), write-only ~**146**, but a **balanced 1R+1W op ~97** — *below either pure
  direction*. Mixing reads+writes ~halves throughput; every pointwise op writes its output, so
  it caps near ~100. **Mechanism OPEN** (read/write turnaround? half-duplex? shared bus upstream
  of DRAM?) — Rung 9 shows shared-bus saturation at ~4 cores (not per-core burst); `aiu-smi`
  bus-utilization is the decider. (Replaces the old "~half peak" guess.)
- **Stream-count BW idea FALSIFIED (Rung 7):** rate is non-monotonic in operand count (gelu 116,
  mul 88, add3 118, add4 124) — 4/5-input fused adds (intermediates staged in LX) match the
  1-input rate. Only plain 2-input `mul`/`add` is anomalously ~15-25% slow (unexplained). Do
  **not** encode a `{2:111,3:80}` table.
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
- **Sizes come from the DEVICE layout (sticks), not the torch logical shape.** Each arg's bytes
  use `FixedTiledLayout.device_layout.device_size` (stick-padded: a row of N fp16 rounds up to
  `ceil(N/64)*64`), available post-`finalize_layouts` where the cost dump runs. Each read is
  sized by ITS OWN buffer's device layout — so a reduction's reduced input is **naturally
  full-sized** (no separate `reduction_size` scaling), and a `[1,N]` broadcast carries its real
  one-row device size (it's excluded anyway). This avoids miscounting e.g. a broadcast operand
  expanded into `[1,N,64]` stick groups.
- **Broadcast/scalar inputs are CACHED → ~free** (Rung 6, VERIFIED). An input whose index
  references fewer loop vars than the output rank — **including 0 vars (a scalar like the `1.0`
  in `x+1.0`)** — is loaded once and reused, so it adds ~no HBM traffic. Excluded from
  `hbm_bytes` (encoded in `cost_model.py`; the dump flags it via `n_index_vars < n_out_vars`).
- **Bulk-load / contiguous assumption.** The model assumes the **default contiguous layout**:
  each core's tile is contiguous and stages in one **bulk DMA read** (not many scattered
  per-stick reads). Spyre's memory requests are limited, so contiguous layout is *required* for
  full bandwidth (`tensors_and_layouts.md`). A strided/scattered layout would move less BW and is
  **NOT modeled** (would need an access-pattern term).
- **`BW_HBM≈111` is a read+write BLEND, not a peak.** It is the balanced-1R+1W rate. Read-heavy
  ops (reductions) stream at ~**176** GB/s and write-heavy at ~**146** (§5.2 Rung 8), so a
  read/write-aware BW is a future refinement; for typical balanced pointwise the single 111 is
  well-calibrated.

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

**GOLDEN measurement = the profiler's kernel device time.** `torch.profiler`
(`ProfilerActivity.PrivateUse1`, needs the kineto-spyre wheel; `examples/profile_ops.py`)
reports a **"Self SPYRE"** column = the TRUE per-kernel (`sdsc_fused_*`) device time. This is
the standard to trust going forward. Cross-check (gelu[512×1024]): kernel **17.3 µs** ≈ our
*traffic* term (18.9 µs) ✓. Our `SPYRE_PROFILE_SYNC` min (~37.9 µs) = kernel + a separate,
**non-deterministic ~20 µs overhead** (the profiler's `Memset (Device)` = host/device setup),
so the old **`fixed ≈ 20 µs` is that OVERHEAD bucket, not kernel cost**. When predicting the
KERNEL time, re-fit `fill_ns` against profiler kernel times across sizes (expect it to drop to a
small device pipeline-fill). The Memset scales (12.5 µs tiny → 28.8 µs at 512×1024) and sits
OUTSIDE our `kernel_timer` bracket — characterize it with a `profile_ops` size sweep.

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

### 5.2 Broadcast, stream-count, and bandwidth (`run_cost_model_plan.sh`, 2026-06-15/16, fp16)

- **Rung 6 (broadcast) — CACHED, general.** `bcast` (`a+b[1,N]`) 36 µs vs `add` (full) 58;
  `mulbcast` 35 vs `mul` 58. The broadcast operand lands on the **2-pass** (no-broadcast)
  latency ⇒ loaded once, ~free; not add-specific. Scalars too (`x+1.0` ≈ `gelu`). **Encoded.**
- **Rung 7 (stream count) — FALSIFIES a stream-count BW law.** Per-byte rate is NON-monotonic in
  operand count: gelu(2-stream) 116, mul(3) 88, add3(4) 118, add4(5) 124. The n-ary adds fuse to
  one kernel **staging intermediates in LX ⇒ all have just 1 HBM write**; extra reads barely
  cost (reads are cheap). So `{2:111,3:80}` is WRONG. Only plain 2-input `mul`/`add` is
  anomalously ~15-25% slow — **UNEXPLAINED** (known residual).
- **Rung 8 (bandwidth, asymptotes at 16k–64k cols).** read-only (`sum`) ~**176** GB/s, write-only
  (broadcast-fill) ~**146**, balanced 1R+1W (`neg`/`copy`) ~**97**. `neg`≈`copy` confirms the
  scalar is free. **Reads ~saturate the 204.8 LPDDR5 peak; mixing in writes ~halves it** — the
  real reason pointwise caps ~100-110 (not a "half the DRAM peak" mystery).
  *Byte-count sanity check:* at the same tensor size `copy` (1R+1W) moves 2× the bytes of `read`
  (~1R), so the bytes are counted consistently. But the TIME ratio is **not** a clean 2× — it is
  1.3× (small, overhead-compressed) → 3.4× (large), i.e. >2× once size dominates. That excess
  over 2× IS the R+W penalty (`copy` runs at ~half `read`'s BW); a clean 2× would mean no penalty.
- **Rung 9 (copy × SENCORES, large fixed size).** BW rises 1→4 cores (41→112) then plateaus
  ~100; **bigger per-core tiles do NOT help** (1 core is slowest) ⇒ the cap is a **shared-bus
  saturation reached at ~4 cores**, NOT per-core burst/scratchpad size.
- **Rung 10 (write-fraction V-curve) — CONFOUNDED.** Multi-output `w2`/`w3` re-read `x` per
  output (not a shared load), so they aren't clean 1R:NW; do not use. `aiu-smi` is the proper way
  to vary the read/write mix — see `bandwidth_turnaround_experiment.md`.

**`mul`/`add` anomaly (open):** the plain 2-input binary runs ~15-25% slower per byte than gelu
or the n-ary fused adds. Not stream-count (add3/add4 are fast), not arithmetic (Rung 2). The
n-ary adds stage their intermediate in LX while `mul` writes straight to HBM — but why that
single difference costs ~20% is unresolved. Flagged, not modeled.

### 5.3 GOLDEN re-anchor on profiler kernel time (`run_profile_sweep.sh`, 2026-06-18, section A)

We re-built the model on the **profiler's per-kernel device time** ("Self SPYRE"), discarding
the old SPYRE_PROFILE_SYNC fit (whose ~20 µs "fixed" was non-deterministic overhead). Section A
(neg + gelu size sweep, fp16):

- **Kernel time is linear in I/O, with ~ZERO fixed.** `kernel = fill + bytes/BW`: neg fill −2.4 µs
  (≈0), BW **104** GB/s, R²=0.9997; gelu fill −3.4 µs (≈0), BW **100**, R²=1.000. So **`fill_ns→0`,
  `BW_HBM→102`** (balanced 1R+1W kernel rate). gelu≈neg ⇒ arithmetic-free on the golden time.
- **The Memset/setup overhead is NOT fixed — it scales with I/O.** neg `11.5 µs + 0.0275 ns/elem`,
  gelu `25.9 µs + 0.0244 ns/elem` (fixed part noisy/non-deterministic; the per-elem scaling is
  real, ~60% of the kernel slope). So the old "20 µs" was just the fixed component at small sizes.

`cost_model.py` updated: `fill_ns=0`, `bw_hbm_gbps=102`. **Pending** (sweep sections B–F not yet
run): re-anchor read/write BW + the R+W penalty (D), reductions `bw_read`/`psum` (F) — those
constants are still MIN-based and flagged in `CostParams`.

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
  (**calibrated `fill_ns=20000, bw_hbm_gbps=111`; LX free; broadcast/scalar args excluded from
  `hbm_bytes`**), `predict_ops`, `predict_op`, `explain`. No torch deps ⇒ path-loadable/testable.
- `torch_spyre/_inductor/dump_cost_model.py` — `extract_features(operations)` over live IR
  (cores from `op_it_space_splits`; per-arg bytes + LX/HBM via allocation propagation). Broadcast
  flag = `n_index_vars < n_out_vars` (**includes 0-var scalars** — fixes counting `x+1.0`'s `1.0`
  as a full read). Hook wired after the AFTER LoopLevel dump; `SPYRE_DUMP_COST=1` prints both.
- `examples/bench_bandwidth.py` — DRAM bandwidth probe: `BENCH_BW_OP=neg|copy|read|write|w2|w3`,
  `BENCH_BW_SUSTAIN_S=N` (saturate for `aiu-smi` sampling). Computes effective BW vs the 204.8
  peak. Companion: `notes/bandwidth_turnaround_experiment.md`.
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
- **DRAM is LPDDR5; aggregate peak = 204.8 GB/s** (`_HBM_BW_GBS` in `work_division.py`; the
  compiler's matmul model uses it with a cohort penalty past 8 cores). Reachable only by
  *unidirectional* streaming — read-only nearly hits it (~176, Rung 8); balanced read+write tops
  out ~half (~97). Naming note: "HBM" in the codebase is legacy; the device DRAM is LPDDR5.
- **Bulk load:** a tile's sticks are stored contiguously, so a whole tile stages in **one bulk
  DMA read** instead of many scattered per-stick reads; memory requests are limited ⇒ contiguous
  stick layout is required for full bandwidth (`tensors_and_layouts.md`).
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
# 6) broadcast reuse: bcast (a+b[1,N]) vs add (full); + mulbcast vs mul  [CACHED/free]
# 7) stream count: gelu/mul/add3/add4 size sweep  [stream-count BW law FALSIFIED]
# 8) bandwidth: neg/copy/read/write size sweep  [read 176 / write 146 / 1R+1W 97]
# 9) tile-size: copy x SENCORES at large size   [shared-bus saturates ~4 cores]
# 10) write-fraction: copy/w2/w3                 [CONFOUNDED — w2/w3 re-read x; ignore]
# 11) reductions: sumrow/amax/mean + sumall      [read@~176 + ring combine; INITIAL]
```

The full, current ladder (rungs 1–10) is `examples/run_cost_model_plan.sh`. The `aiu-smi`
mechanism capture is a separate two-terminal run — see `bandwidth_turnaround_experiment.md`.

---

## 9. Verification checklist (earn the right to stay simple)

- [x] effective `BW_HBM` constant across shapes (size-sweep linearity) — rung 1 ✓ (~111 GB/s)
- [x] `fixed` is a real op-independent intercept — rung 1 ✓ (~20 µs; rung 2 confirms op-indep)
- [x] BW shared vs per-core — rung 5 ✓ (SHARED; flat ≥2 cores; cores not a direct term)
- [x] traffic = Σ inputs + output — rung 3: holds for 1-in; **plain 2-in `mul`/`add` ~20% over,
  UNEXPLAINED** (rung 7 ruled out a stream-count law; n-ary fused adds are NOT slow)
- [x] arithmetic free for pointwise (relu == gelu?) — rung 2 ✓
- [x] **broadcast** reuse: cached vs re-fetched — rung 6 ✓ (**CACHED/~free**, incl. scalars; encoded)
- [x] **LX cost** from chain-depth — rung 4 → per-op LX cost below noise ⇒ **LX treated as ~free** (term dropped)
- [x] cost-model `extract_features` matches the IR — confirmed (`SPYRE_DUMP_COST` op counts/bytes)
- [x] **stream count → BW?** — rung 7 ✓ **FALSIFIED** (non-monotonic; do not encode a per-stream table)
- [x] **read vs write vs balanced BW** — rung 8 ✓ (read ~176, write ~146, 1R+1W ~97)
- [ ] **mechanism of the read+write penalty** (turnaround / half-duplex / shared bus) — needs `aiu-smi`

---

## 10. Plan / next steps

1. ~~Rungs 1–10~~ DONE — `fixed≈20µs`, `BW_HBM≈111` (balanced R+W); arithmetic-free, shared-BW,
   LX-free, **broadcast/scalar cached (encoded)**, stream-count law **falsified**, and the
   bandwidth picture (read 176 / write 146 / 1R+1W 97) all verified.
2. **`aiu-smi` capture** (copy/read/write/neg) → resolve the **mechanism** of the read+write
   penalty (turnaround vs half-duplex vs shared-bus) and quantify it. See
   `bandwidth_turnaround_experiment.md`. THE open experimental item.
3. Decide whether to go **read/write-aware** in the model (reads ~176, writes ~146) — only worth
   it if a balanced-blend (111) misranks read-heavy ops; reductions are the first such case.
4. Run down the **`mul`/`add` ~20% anomaly** (plain 2-input binary) — or accept it as a residual.
5. **Reductions — INITIAL model BUILT** (`cost_model.py` + `dump_cost_model.py`): read the FULL
   input (`out_elems × reduction_size`, from `Reduction.get_reduction_size()`) at the read rate
   (~176), write the small output, **+ a `(k−1)·out_elems·psum` ring-combine** when the reduced
   axis is split across `k` cores (mirrors the matmul PSUM, `_PSUM_PER_ELEM_US=1.4e-4 µs/elem`).
   Predicts `sum(dim=-1)[512×16384]` at ~115 µs vs measured ~118 (~2%). **Rung 11 calibrates** it
   (arithmetic-free, read-BW, combine). Then re-validate on `mean`/softmax.
6. Later: tile-fusion regime (hinted examples), matmul (`pt` unit / compute-bound), the LX
   capacity cliff as a hard constraint, and an access-pattern term if non-contiguous layouts matter.

---

## 11. Open questions

RESOLVED: shared-vs-per-core HBM BW (rung 5 → SHARED); LX is **~free** (rung 4; term dropped,
~29× HBM); arithmetic-free (rung 2); **broadcast/scalar inputs CACHED → ~free** (rung 6, encoded);
**stream-count BW law FALSIFIED** (rung 7 — non-monotonic; no per-stream table); and the
**`~111` cap is a read+write blend** (rung 8: read ~176 ≈ 86% of the 204.8 LPDDR5 peak, write
~146, balanced 1R+1W ~97 — mixing reads+writes ~halves throughput, NOT a mysterious "half-peak").

Still open:
- **Mechanism of the read+write penalty** — *why* does 1R+1W (~97) run at half of read-only
  (~176)? Candidates: DRAM read/write **turnaround**; **half-duplex** link; **shared bus**
  saturation upstream of DRAM (rung 9: copy saturates at ~4 cores, NOT per-core burst — bigger
  tiles don't help). DECIDER: `aiu-smi` DDR bandwidth + bus-utilization during copy/read/write
  (idle bus ⇒ turnaround; busy-but-slow ⇒ controller limit). See
  `bandwidth_turnaround_experiment.md`. **THE open item.**
- **`mul`/`add` ~20% anomaly** — plain 2-input binary slower per byte than gelu / n-ary fused
  adds; mechanism unknown (only structural diff: it writes to HBM vs the adds' LX intermediate).
- **Read/write-aware BW?** — reductions read at ~176, not the 111 blend; only refine if the blend
  misranks them.
- **Reduction model calibration (rung 11):** an INITIAL reduction model is built (read full input
  @ read rate + ring combine). Open: confirm arithmetic-free for reductions; fit the reduction
  read rate (is it the ~176 read asymptote?); calibrate the ring-combine `psum_per_elem` (1.4e-4
  µs/elem is the matmul starting guess) — expected negligible vs HBM I/O until compute-bound. Also
  the `reduction_cores` (k) extraction is a heuristic (`out_elems < cores`) — refine if it matters.
- LX precise BW unresolvable here (signal < noise) — revisit only with a larger-tile LX sweep.
- **Access-pattern / bulk-load:** does a strided (non-contiguous) layout drop BW vs the modeled
  contiguous case? (Exp A in the bandwidth note; gated on confirming the compiled path honors a
  custom layout.)
- **Re-anchor on the profiler kernel time (golden).** Collect `profile_ops` "Self SPYRE" kernel
  times across sizes/ops and **re-fit `fill_ns`** — the old ~20 µs was the non-deterministic
  host/Memset overhead, not kernel cost; the kernel-time fixed should be small. Also characterize
  the `Memset (Device)` (scales 12.5→28.8 µs; is it per-call work we should surface?).
- `fixed`'s ~7 µs host residue: stable across kernels / would a device-only timer change it?
