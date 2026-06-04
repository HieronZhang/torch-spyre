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

"""Opt-in host-side timing of Spyre kernel launches.

Enabled by ``SPYRE_PROFILE=1``. ``kernel_timer`` wraps the device launch in
``SpyreSDSCKernelRunner.run`` and accumulates per-kernel wall-clock; a summary
is printed at process exit (to stderr, or to ``SPYRE_PROFILE_FILE``). A no-op
unless the env var is set.

CAVEAT: device execution is asynchronous (``executeProgramAsync``) and the
stream-sync hooks are no-ops, so this measures host-side *dispatch* latency
unless a synchronization (e.g. copying a dependent output to host) forces
device completion. For true device-inclusive latency, time an end-to-end
region that ends in ``.cpu()`` -- see :func:`torch_spyre.execution.bench.measure_latency`
-- ideally on a single-kernel program. Conversely, if the per-kernel time here
*scales with problem size*, the runtime is effectively synchronous and these
numbers are already device-inclusive (a useful probe in itself).

Because the device is a static dataflow engine, per-kernel latency is
deterministic; the reported ``min`` strips host-side jitter and is the value to
trust.
"""

import atexit
import contextlib
import math
import os
import sys
import time

_TRUTHY = {"1", "true", "yes", "on"}

# name -> {count, total_ns, min_ns, max_ns}
_records: dict[str, dict] = {}
_atexit_registered = False


def profile_enabled() -> bool:
    """Return True when SPYRE_PROFILE requests kernel timing."""
    return os.environ.get("SPYRE_PROFILE", "").strip().lower() in _TRUTHY


def reset() -> None:
    """Clear accumulated records (useful for tests / between benchmark phases)."""
    _records.clear()


def records() -> dict[str, dict]:
    """Return the raw per-kernel timing records."""
    return _records


@contextlib.contextmanager
def kernel_timer(name: str):
    """Time the wrapped device launch and accumulate it under ``name``.

    No-op (zero overhead beyond the env check) unless SPYRE_PROFILE is set.
    """
    if not profile_enabled():
        yield
        return
    _ensure_atexit()
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed = time.perf_counter_ns() - start
        rec = _records.get(name)
        if rec is None:
            rec = {"count": 0, "total_ns": 0, "min_ns": math.inf, "max_ns": 0}
            _records[name] = rec
        rec["count"] += 1
        rec["total_ns"] += elapsed
        rec["min_ns"] = min(rec["min_ns"], elapsed)
        rec["max_ns"] = max(rec["max_ns"], elapsed)


def format_report() -> str:
    """Render the accumulated per-kernel timings as a text table."""
    if not _records:
        return "[SPYRE_PROFILE] no kernel launches recorded"
    lines = [
        "==== SPYRE_PROFILE: per-kernel host launch timing "
        "(min strips host jitter) ====",
        f"{'kernel':<40}{'count':>7}{'min_us':>11}{'mean_us':>11}{'max_us':>11}",
    ]
    for name in sorted(_records):
        rec = _records[name]
        count = rec["count"]
        min_us = rec["min_ns"] / 1000.0
        mean_us = (rec["total_ns"] / count) / 1000.0 if count else 0.0
        max_us = rec["max_ns"] / 1000.0
        lines.append(
            f"{name:<40}{count:>7}{min_us:>11.3f}{mean_us:>11.3f}{max_us:>11.3f}"
        )
    lines.append(
        "[note] host-side dispatch unless a sync forced device completion; "
        "see bench.measure_latency"
    )
    return "\n".join(lines)


def report() -> None:
    """Emit the timing report to SPYRE_PROFILE_FILE or stderr."""
    text = format_report()
    dest = os.environ.get("SPYRE_PROFILE_FILE")
    if dest:
        with open(dest, "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        sys.stderr.write(text)
        sys.stderr.write("\n")
        sys.stderr.flush()


def _ensure_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(report)
        _atexit_registered = True
