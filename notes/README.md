# Spyre cost-model tooling

Three additive features we built on top of the torch-spyre compiler. Each is a
no-op unless its env var is set, so normal runs are unaffected. The example
commands use the profiler harness
[docs/source/user_guide/examples/profile_ops.py](../docs/source/user_guide/examples/profile_ops.py)
(knobs: `BENCH_OP`, `BENCH_ROWS`, `BENCH_COLS`, `BENCH_TILES`, `SENCORES`,
`LX_PLANNING`).

> **Note:** the dumps and cost-model prediction only fire on a fresh compile, so
> clear the TorchInductor cache before every run — otherwise a cached graph is
> reused and nothing is dumped. The reliable way is to delete the on-disk cache
> (`rm -rf /tmp/torchinductor_*`).

## 1. Dump FX graph and loop-level IR

`SPYRE_DUMP_IR=1` prints the post-grad ATen FX graph and the after-pre-scheduling
LoopLevel IR at compile time (to stderr, or to `SPYRE_DUMP_IR_FILE`). The
LoopLevel dump also shows coarse-tiling metadata — `dim_hints`, `loop_info`
(`loop_count` / `loop_tiled_reduction_dims`), `op_it_space_splits` — when a
`spyre_hint` is present.

- Example FX-graph output: [haoyang_logs/fx_graph_dump.log](../haoyang_logs/fx_graph_dump.log)
- Example LoopLevel-IR output: [haoyang_logs/loop_ir_dump.log](../haoyang_logs/loop_ir_dump.log)
- Coarse-tiled reduction IR (fill + K×(reduce,combine)):
  [haoyang_logs/coarse_tile_sum.log](../haoyang_logs/coarse_tile_sum.log)

```bash
SPYRE_DUMP_IR=1 BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS=1024 \
    python docs/source/user_guide/examples/profile_ops.py
```

## 2. Per-kernel device time via the PyTorch profiler 

The golden measurement is the **`torch.profiler` "Self SPYRE" per-kernel device
time** of each `sdsc_fused_*` kernel (needs the kineto-spyre wheel: [docs/source/user_guide/profiling/pytorch_profiler.md](../docs/source/user_guide/profiling/pytorch_profiler.md)).
The separate **"Memset (Device)"** event is non-deterministic, size-scaling
host/setup overhead — reported, but NOT kernel time. 

Our `profile_ops.py`
emits a parseable `SUMMARY … kernel_us=… pred_us=… bw_gbps=…` line per run;
[profile_test.py](../profile_test.py) is a minimal standalone demo that prints the
`sdsc_fused` kernel time next to the cost-model prediction.

- Example profiled output (kernel time + prediction per op):
  [haoyang_logs/grand_sweep_20260619_151132.log](../haoyang_logs/grand_sweep_20260619_151132.log)

```bash
# one op: prints the profiler table + a SUMMARY with kernel_us / bw_gbps
BENCH_OP=gelu BENCH_ROWS=2048 BENCH_COLS=4096 \
    python docs/source/user_guide/examples/profile_ops.py
# minimal standalone profiler demo (kernel time vs cost model):
python profile_test.py
```

## 3. Cost model

`SPYRE_DUMP_COST=1` extracts cost features from the after-pre-scheduling LoopLevel
IR and prints, at compile time, the **per-tensor device-layout I/O**
(dims · residency · byte calc · `hbm counted` / `xL` loop factor) and the
**step-by-step prediction**:

```
T = fill + (R + W) / BW_PEAK + α·min(R, W) + c_loop·L
```

a single peak bandwidth (`BW_PEAK≈150 GB/s`) minus a read/write **turnaround**
penalty on the overlap `min(R,W)` (`α≈0.0057 ns/B`), plus a coarse-tiling
per-tile loop cost (`c_loop≈860 ns/tile`, `L` = tile trip count). LX-resident and
broadcast operands are counted once / ~free. 
```bash
SPYRE_DUMP_COST=1 BENCH_OP=gelu BENCH_ROWS=2048 BENCH_COLS=4096 \
    python docs/source/user_guide/examples/profile_ops.py
```

### Reduction and broadcast kernels

The harness covers reductions and broadcasts so the cost model can be checked on
read-dominated and cached-operand traffic:

```bash
# REDUCTIONS (read-dominated): sumrow=dim-1, sumcol=dim-0, amax/mean, read
BENCH_OP=sumrow BENCH_ROWS=2048 BENCH_COLS=4096 \
    python docs/source/user_guide/examples/profile_ops.py
# BROADCAST (operand loaded once): bcast=a[R,C]+b[1,C], bcastcol=a[R,C]+b[R,1]
BENCH_OP=bcast  BENCH_ROWS=2048 BENCH_COLS=4096 \
    python docs/source/user_guide/examples/profile_ops.py
# COARSE-TILED dim0 reduction (loop-aware: fill + K×(reduce,combine)); LX on/off
BENCH_OP=ctsum BENCH_TILES=8 LX_PLANNING=0 BENCH_ROWS=2048 BENCH_COLS=512 \
    python docs/source/user_guide/examples/profile_ops.py
```

## One challenge: Mem access pattern changes effective bandwidth

Two kernels can stream the **same** input yet hit very different DRAM bandwidth —
the cost depends on the read/write *mix*, not the byte count alone. Run a
**read-dominated** reduction and a **balanced** pointwise op on the same
`[2048, 2048]` fp16 tensor, pinned to all 32 cores (`SENCORES=32`; the 2048 output
rows give every core 64 independent rows for the reduction, and the full grid for
the pointwise — so both fully utilize the device). Each command prints, in order,
the **loop-level IR**, the **device-layout I/O calculation**, and the profiler's
**`sdsc_fused` kernel time** (clear the inductor cache first — see the note at the
top):

```bash
# read = x.sum(dim=-1): reduce the 2048 cols to a [2048] vector
#   -> reads the full input once, writes an almost-empty output  (READ-DOMINATED)
SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=read BENCH_ROWS=2048 BENCH_COLS=2048 \
    python docs/source/user_guide/examples/profile_ops.py
# neg = -x: one full read + one full write of the same shape  (1R + 1W, BALANCED)
SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=neg  BENCH_ROWS=2048 BENCH_COLS=2048 \
    python docs/source/user_guide/examples/profile_ops.py
```

Both commands read the **same full input**; only the write differs. `read` writes
an almost-empty output, so its traffic is **one-directional** and the `bw_gbps` in
its `SUMMARY` lands near the ~150 GB/s peak. `neg` writes a full output, so reads
and writes must **interleave on the shared HBM bus**, which keeps turning around
between the two directions — its `bw_gbps` collapses well below the peak, *below*
either pure direction. Same input, same compute, but the **write half of the
access pattern costs a read/write turnaround penalty** that nearly halves the
effective DRAM bandwidth. (This is the `α·min(R,W)` term in §3: the read/write
overlap `min(R,W)` is ≈0 for `read` and maximal for `neg`.)
