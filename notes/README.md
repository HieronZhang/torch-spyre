# Spyre cost-model tooling 

Three additive features we built on top of the torch-spyre compiler. Each is a
no-op unless its env var is set, so normal runs are unaffected.

> **Note:** the dumps and cost-model prediction only fire on a fresh compile, so
> clear the TorchInductor cache before every run (e.g. set
> `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`) — otherwise a cached graph is reused and
> nothing is dumped.

## 1. Dump FX graph and loop-level IR

Prints the post-grad ATen FX graph and the after-pre-scheduling LoopLevel IR at
compile time, to stderr or to `SPYRE_DUMP_IR_FILE`.

- Example FX-graph output: [haoyang_logs/fx_graph_dump.log](../haoyang_logs/fx_graph_dump.log)
- Example LoopLevel-IR output: [haoyang_logs/loop_ir_dump.log](../haoyang_logs/loop_ir_dump.log)

```bash
SPYRE_DUMP_IR=1 BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS=1024 python examples/bench_ops.py
```

## 2. Per-op device time measurement

Reports the deterministic per-kernel device latency (min over N runs) by
synchronizing the device after each launch.

- How/where the timers are placed: [timing_measurement.md](timing_measurement.md)
- Example measured output: [haoyang_logs/example_lx.log](../haoyang_logs/example_lx.log)

```bash
SPYRE_PROFILE=1 SPYRE_PROFILE_SYNC=1 BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS=1024 python examples/bench_ops.py
```

## 3. Pointwise-op cost model

Predicts relative device latency of a pointwise op/bundle from its LoopLevel IR
(`T = fill + HBM_bytes / BW_HBM`, LX traffic ~free).

- Presentation of the model with figures: [cost_model_demo.ipynb](cost_model_demo.ipynb)
- Full prediction-vs-measured calibration log: [cost_model_results_20260612_173311.log](../haoyang_logs/cost_model_results_20260612_173311.log)

```bash
SPYRE_DUMP_COST=1 BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS=1024 python examples/bench_ops.py
```
