# How we measure per-op device time
We measure the **host wall-clock time** between two host↔device
synchronizations, run the op a fixed number of times in between, and divide by that count —
giving the per-execution device latency (the syncs guarantee the device actually finished the
timed work). We updated this method after the 06/10 merge, which exposed a device-sync
primitive (`_C.synchronize()`): it lets us sync **per launch without a D2H copy**, so we can
read clean **per-op** device latency that the old end-to-end `.cpu()` approach could not
resolve.

The **current** per-kernel
device-sync method, and the **old** `.cpu()` end-to-end method it replaced. (Companion to
`cost_model_design.md`.)

## At a glance

| | Current — `measure_device` | Old — `measure_latency` |
|---|---|---|
| sync point | `_C.synchronize()` **per launch** | `.cpu()` (D2H copy) per sample |
| what is timed | one kernel (device-inclusive) | whole compiled call (host + device + D2H) |
| **executions / measurement** | **1** | **`inner` (=400)**, then ÷ inner |
| granularity | per SDSC bundle (= per op) | end-to-end |
| enabled by | `SPYRE_PROFILE_SYNC=1` | always (only option pre-merge) |

Both report the **min over many samples**: Spyre is static-dataflow ⇒ device latency is
deterministic; host/OS jitter only *adds* time, so the minimum is the true latency.

## Current method — per-kernel device sync

A `perf_counter_ns()` timer brackets the device launch inside `SpyreSDSCKernelRunner.run`
([kernel_runner.py:44](../torch_spyre/execution/kernel_runner.py#L44)); with
`SPYRE_PROFILE_SYNC=1`, `kernel_timer`
([profiling.py:96](../torch_spyre/execution/profiling.py#L96)) calls `_C.synchronize()` in its
`finally`, *before* stopping the clock:

```python
with kernel_timer(self.kernel_name):       # kernel_runner.py
    launch_kernel(self.code_dir, args)     # async launch
# kernel_timer:  start = perf_counter_ns(); yield (launch runs);
#                if sync: _device_synchronize(); elapsed = perf_counter_ns() - start
```

- **Why the sync:** the launch (`executeProgramAsync`,
  [spyre_stream.cpp:217](../torch_spyre/csrc/spyre_stream.cpp#L217)) is asynchronous and
  returns before the device finishes. The sync (`handle->synchronize()`,
  [spyre_stream.cpp:128](../torch_spyre/csrc/spyre_stream.cpp#L128)) makes the bracketed region
  include the device execution. Without it, the timer sees only the ~7 µs host dispatch.
- **Bracketed region:** `[host dispatch ~7µs] + [device kernel] + [host sync return]` →
  device time + a small (~7 µs) op-independent host residue.
- **Executions per measurement = 1.** Each launch self-syncs, so one execution is already one
  clean device-latency sample — no amortization needed.
- **Driver** `measure_device` ([bench.py:173](../torch_spyre/execution/bench.py#L173)):
  `warmup` launches → `profiling.reset()` → `runs` (=100) launches → report the per-kernel
  **`min_ns`**.
- **Granularity:** one `run()` = one SDSC bundle. Single-op example → that op; a fused
  multi-op kernel (e.g. softmax = 5 ops in `sdsc_fused__softmax_0`) → one number for the whole
  bundle.

## Old method — `.cpu()` end-to-end sync

Before the merge added `_C.synchronize()`, there was **no Python-callable device wait** (the
c10 stream-sync hooks were no-op stubs and `elapsedTime` returned `0`). The only way to force
completion was to **copy a dependent output to host with `.cpu()`** — the runtime serializes
that device→host (D2H) transfer, so it can't return until the data is ready.

`measure_latency` ([bench.py:116](../torch_spyre/execution/bench.py#L116)) timed the **whole
compiled call** and ended each sample with `.cpu()` (`_default_sync`,
[bench.py:38](../torch_spyre/execution/bench.py#L38)). Because one `.cpu()` (full drain +
~1 MB D2H ≈ 1 ms) dwarfs a µs-scale kernel, it fired **`inner` (=400) launches per sample**,
synced **once**, and divided by `inner`.

- **Executions per measurement = `inner` (=400)** — to amortize the costly `.cpu()`.
- **Limitations:** times end-to-end, not the kernel; the ~70 µs host-per-call floor hid
  small-op device compute (small ops gave no usable signal); drains only because the *output*
  is copied, so no per-kernel number for an internal kernel.

`.cpu()` remains `measure_latency`'s default — it's the right tool for true **user-visible**
latency (which legitimately includes the D2H transfer). `bench.device_sync`
([bench.py:52](../torch_spyre/execution/bench.py#L52)) is the no-copy `_C.synchronize()`
alternative.

## Env vars

| var | effect |
|---|---|
| `SPYRE_PROFILE=1` | enable the per-kernel timer + the at-exit report |
| `SPYRE_PROFILE_SYNC=1` | sync the device after each launch → per-kernel **device** latency |
| `SPYRE_PROFILE_FILE=path` | write the report to a file instead of stderr |

Without `SPYRE_PROFILE_SYNC`, the same timer reports host **dispatch** latency (~7 µs) only.
