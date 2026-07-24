#!/usr/bin/env python3
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
"""Reproduce the §3 chained-adds result directly ON THE SPYRE DEVICE.

Self-contained: it runs the add-chain workloads on Spyre, profiles the on-device kernel time,
and computes the excess-cost metric itself -- it does NOT read the recorded sweep data and does
NOT import profile_ops.py. Run it on the machine with the Spyre accelerator:

    python3 docs/source/user_guide/examples/repro_sec3_addchain.py

Knobs (env): ROWS (default 2048), COLS_LIST ("1024,4096,16384"), WARMUP (5), REPS (7).

The §3 result it reproduces: an n-input sum ``a + b + c + …`` compiles to a chain of dependent
binary adds. The FUSED chain (``add_n``, one kernel) and the SEPARATE control (``add_n_sep``, each
``+`` its own kernel, same read-after-write dependency, no fusion) cost the same at every chain
length EXCEPT add4, where they diverge by ~+0.31 single-``add``s. The metric is

    excess(add_n) = mean over shapes of [ t(add_n) / t(add) - (n - 1) ]

("extra single-adds of time beyond the pure byte count"). Scratchpad planning is forced OFF
(``LX_PLANNING=0``) so every intermediate round-trips HBM -- the regime the §3 figure analyses.
"""
import os
import statistics

# Set BEFORE importing torch/torch_spyre: force a real compile and the scratchpad-OFF regime.
os.environ.setdefault("LX_PLANNING", "0")  # intermediates spill to HBM (the fig3b regime)
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
import torch_spyre  # noqa: E402,F401  -- importing registers the "spyre" PrivateUse1 device
from torch.profiler import ProfilerActivity, profile  # noqa: E402

DEVICE = torch.device("spyre")
ROWS = int(os.environ.get("ROWS", "2048"))
COLS_LIST = [int(c) for c in os.environ.get("COLS_LIST", "1024,4096,16384").split(",")]
WARMUP = int(os.environ.get("WARMUP", "5"))
REPS = max(1, int(os.environ.get("REPS", "7")))
torch.manual_seed(0xAFFE)


def _rand(rows, cols):
    return torch.rand(rows, cols, dtype=torch.float16).to(DEVICE)


def _sync(out):  # move result(s) to host so the device actually finishes the work
    for t in out if isinstance(out, (tuple, list)) else (out,):
        t.cpu()


def measure_us(fn, args):
    """Median on-device kernel time (µs) over REPS profiled traces. Kernel = the fused compute
    event(s); Memset/Memcpy are excluded -- exactly the classification profile_ops.py uses. For a
    separate-kernel chain the profiler sums ALL its sub-kernels, so this is the whole chain's time."""
    for _ in range(WARMUP):  # compile + warm the kernel(s)
        _sync(fn(*args))
    kernels = []
    for _ in range(REPS):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            _sync(fn(*args))
        k = 0.0
        for ev in prof.key_averages():
            us = getattr(ev, "self_device_time_total", 0) or 0
            key = ev.key or ""
            if us and us > 0 and "Memset" not in key and "Memcpy" not in key:
                k += us
        if k > 0:
            kernels.append(k)
    return statistics.median(kernels) if kernels else float("nan")


# ---- the workloads, built here (NOT imported from profile_ops.py) ----
def _fused_sum(*ts):  # add_n: one compiled graph -> the chain fuses into a single kernel
    acc = ts[0]
    for t in ts[1:]:
        acc = acc + t
    return acc


_add = torch.compile(lambda a, b: a + b)  # one reusable compiled binary add


def _separate_chain(*ts):  # add_n_sep: same left-assoc chain, each '+' its OWN kernel launch
    acc = _add(ts[0], ts[1])  # (plain Python orchestration -> no fusion across the adds)
    for t in ts[2:]:
        acc = _add(acc, t)
    return acc


def run_shape(rows, cols):
    """{op -> kernel_us} for `add` (baseline) and add3..add6 fused + add3_sep..add6_sep."""
    out = {"add": measure_us(_add, (_rand(rows, cols), _rand(rows, cols)))}
    for n in (3, 4, 5, 6):
        fused = torch.compile(_fused_sum)  # fresh graph per arity (mirrors a per-op run)
        out[f"add{n}"] = measure_us(fused, tuple(_rand(rows, cols) for _ in range(n)))
        out[f"add{n}_sep"] = measure_us(
            _separate_chain, tuple(_rand(rows, cols) for _ in range(n))
        )
    return out


def main():
    per_shape = {c: run_shape(ROWS, c) for c in COLS_LIST}

    print("\n§3 chained adds on Spyre — excess = t(add_n)/t(add) − (n−1)")
    print(f"ROWS={ROWS}  LX_PLANNING=0  COLS={COLS_LIST}  (warmup={WARMUP}, reps={REPS})\n")
    print("raw median kernel µs:")
    for c in COLS_LIST:
        s = per_shape[c]
        print(
            f"  COLS={c:>6}: "
            + "  ".join(f"{op}={s[op]:.0f}" for op in ["add", "add3", "add3_sep", "add4", "add4_sep"])
        )
    print(f"\n{'reads':>5} {'op':>6} | {'fused':>7} | {'separate':>8} | {'gap':>7}")
    print("-" * 42)
    for n in (3, 4, 5, 6):
        fe = statistics.mean(per_shape[c][f"add{n}"] / per_shape[c]["add"] - (n - 1) for c in COLS_LIST)
        se = statistics.mean(
            per_shape[c][f"add{n}_sep"] / per_shape[c]["add"] - (n - 1) for c in COLS_LIST
        )
        gap = fe - se
        flag = "  <-- the add4 divergence" if abs(gap) > 0.1 else ""
        print(f"{n - 2:>5} {'add' + str(n):>6} | {fe:>+7.3f} | {se:>+8.3f} | {gap:>+7.3f}{flag}")
    print("\nExpected: gap ~0 at add3/add5/add6, and ~+0.3 at add4 (fused ≫ separate).")


if __name__ == "__main__":
    main()
