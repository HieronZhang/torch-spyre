# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal latency-measurement harness for a compiled Spyre kernel.

Times compiled softmax end-to-end (min over N runs, forced ``.cpu()`` sync) and
reports the spread so determinism can be confirmed. Run with ``SPYRE_PROFILE=1``
to additionally get the per-kernel launch table at exit.

Example:
    SPYRE_PROFILE=1 python examples/bench_softmax.py
    BENCH_RUNS=50 BENCH_ROWS=512 BENCH_COLS=1024 python examples/bench_softmax.py
"""

import os

import torch

from torch_spyre.execution.bench import device_sync, measure_latency, net_latency_us

DEVICE = torch.device("spyre")
RUNS = int(os.environ.get("BENCH_RUNS", "100"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
# BENCH_SYNC=device drains the device via torch_spyre._C.synchronize (no D2H copy);
# anything else uses the default .cpu() sync.
SYNC = device_sync if os.environ.get("BENCH_SYNC") == "device" else None
# inner: launches per timed sample, synced once -- amortizes the ~ms .cpu() floor
# so the per-launch device cost can surface. 400 was where softmax converged to a
# deterministic (<3% spread) per-call min on hardware; raise it if spread is large.
INNER = int(os.environ.get("BENCH_INNER", "400"))
# Default to a larger softmax so device compute is less dominated by the ~70us
# host-per-call floor. Scale up further (e.g. 8192) to push toward compute-bound.
ROWS = int(os.environ.get("BENCH_ROWS", "4096"))
COLS = int(os.environ.get("BENCH_COLS", "4096"))

torch.manual_seed(0xAFFE)
x = torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE)

compiled_sm = torch.compile(lambda a: torch.softmax(a, dim=0))
# An identity baseline to subtract fixed launch + host/device transfer cost.
compiled_id = torch.compile(lambda a: a + 0.0)

softmax = measure_latency(
    lambda: compiled_sm(x),
    runs=RUNS,
    warmup=WARMUP,
    inner=INNER,
    sync=SYNC,
    label=f"softmax[{ROWS}x{COLS}]",
)
baseline = measure_latency(
    lambda: compiled_id(x),
    runs=RUNS,
    warmup=WARMUP,
    inner=INNER,
    sync=SYNC,
    label=f"identity[{ROWS}x{COLS}]",
)

print(softmax)
print(baseline)
print(
    f"net softmax compute (min - baseline_min): "
    f"{net_latency_us(softmax, baseline):.3f} us"
)
