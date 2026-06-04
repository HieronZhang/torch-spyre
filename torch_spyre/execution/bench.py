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

"""End-to-end latency measurement for Spyre workloads (host wall-clock).

Spyre is a static dataflow engine, so device latency is deterministic; the
only variation in a host-side wall-clock measurement is host/OS/launch jitter.
We therefore take the **min over N runs** (the min strips jitter; the device
portion is constant) and report the spread so determinism can be confirmed.

Because device launch is asynchronous, the timed region must end in a
synchronizing sink so it actually includes device compute. By default we copy
the workload's output to host (``.cpu()``), which serializes. For a clean
*per-kernel* number, run a single-kernel program (e.g. a one-op example, or set
``SPYRE_INDUCTOR_ENABLE_FUSION=0``) so the whole measured region is that kernel,
and subtract an identity/no-op baseline to remove fixed launch + transfer cost.

This module has no torch-spyre internal dependencies and imports torch lazily,
so it can be used as a plain measurement utility.
"""

import statistics
import time
from dataclasses import dataclass


def _default_sync(out) -> None:
    """Force device completion by moving any returned tensors to host."""
    try:
        import torch
    except ImportError:
        return
    if isinstance(out, torch.Tensor):
        out.to("cpu")
    elif isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, torch.Tensor):
                item.to("cpu")


@dataclass
class LatencyStats:
    """Latency samples (nanoseconds) and deterministic-friendly summaries."""

    samples_ns: list[int]
    label: str = ""

    @property
    def min_ns(self) -> int:
        return min(self.samples_ns)

    @property
    def median_ns(self) -> float:
        return statistics.median(self.samples_ns)

    @property
    def max_ns(self) -> int:
        return max(self.samples_ns)

    @property
    def spread(self) -> float:
        """(max - min) / min -- should be ~0 for deterministic device latency."""
        return (self.max_ns - self.min_ns) / self.min_ns if self.min_ns else 0.0

    def summary(self) -> dict:
        return {
            "label": self.label,
            "runs": len(self.samples_ns),
            "min_us": self.min_ns / 1000.0,
            "median_us": self.median_ns / 1000.0,
            "max_us": self.max_ns / 1000.0,
            "spread": self.spread,
        }

    def __str__(self) -> str:
        s = self.summary()
        verdict = "deterministic" if self.spread < 0.05 else "NOISY (host jitter?)"
        return (
            f"{s['label'] or 'latency'}: "
            f"min={s['min_us']:.3f}us median={s['median_us']:.3f}us "
            f"max={s['max_us']:.3f}us  spread={s['spread'] * 100:.1f}%  "
            f"[{verdict}]  (n={s['runs']})"
        )


def measure_latency(
    fn,
    runs: int = 30,
    warmup: int = 3,
    sync=_default_sync,
    label: str = "",
) -> LatencyStats:
    """Time a zero-arg workload ``fn`` end-to-end, min-of-N with a forced sync.

    Args:
        fn: zero-arg callable that runs the workload and returns its output
            (a tensor or sequence of tensors, used for the default sync).
        runs: number of timed iterations (the reported latency is their min).
        warmup: untimed iterations first (absorbs compile + first-launch cost).
        sync: callable applied to ``fn``'s return value to force device
            completion before stopping the clock. Default copies tensors to host.
        label: name for the report line.

    Returns:
        LatencyStats over ``runs`` samples.
    """
    for _ in range(warmup):
        sync(fn())
    samples_ns: list[int] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        out = fn()
        sync(out)
        samples_ns.append(time.perf_counter_ns() - start)
    return LatencyStats(samples_ns, label=label)


def net_latency_us(kernel: LatencyStats, baseline: LatencyStats) -> float:
    """Min kernel latency with the baseline (launch + transfer) min subtracted."""
    return (kernel.min_ns - baseline.min_ns) / 1000.0
