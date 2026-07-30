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
"""Reproduce the §3 add3 bar chart (fig3) directly ON THE SPYRE DEVICE.

Self-contained: it runs each workload on Spyre, prints the FULL torch profiler table for it, and
finally prints a summary of each op's kernel time. It does NOT read the recorded sweep data and
does NOT import profile_ops.py.

    python3 docs/source/user_guide/examples/repro_sec3_addchain.py     # on the Spyre machine

Knobs (env): ROWS (default 2048), COLS (4096), WARMUP (5), REPS (7).

THE OPS (each is an explicit Python program below; all on ROWS x COLS float16 tensors):

  add        = a + b                 -- the UNIT (2R:1W). The byte-count baseline is 2 x this.
  add_indep2 = (a+b, c+d)            -- two INDEPENDENT adds: same 4R:2W bytes as add3, NO
                                        read-after-write dependency  -> lands ON the baseline.
  add3_sep   = add(add(a,b), c)      -- the add3 chain as SEPARATE kernels (each `+` its own
                                        compiled kernel; the intermediate a+b round-trips HBM)
                                        -> ~+7% (the dependency cost).
  add3       = (a+b)+c  [FUSED]       -- the whole chain in ONE compiled kernel, scratchpad OFF
                                        -> ~+7% (== add3_sep, so fusion itself is FREE).
  add3 LX-on = (a+b)+c  [FUSED]       -- same fused chain, scratchpad ON: the intermediate stays
                                        on-chip, the HBM round-trip is gone -> ~-34%.
"""
import os
import statistics

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")  # force a real compile each time

import torch  # noqa: E402
import torch_spyre  # noqa: E402,F401  -- importing registers the "spyre" PrivateUse1 device
import torch_spyre._inductor.config as spyre_config  # noqa: E402  -- toggles scratchpad planning
from torch.profiler import ProfilerActivity, profile  # noqa: E402

DEVICE = torch.device("spyre")
ROWS = int(os.environ.get("ROWS", "2048"))
COLS = int(os.environ.get("COLS", "4096"))
WARMUP = int(os.environ.get("WARMUP", "5"))
REPS = max(1, int(os.environ.get("REPS", "7")))
torch.manual_seed(0xAFFE)


def inputs(n):
    """`n` fresh random ROWS x COLS float16 tensors placed on the Spyre device."""
    return tuple(torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE) for _ in range(n))


# ============================================================================
# THE PROGRAMS -- exactly what each op computes. Plain Python of device tensors;
# `torch.compile` turns each into Spyre kernel(s).
# ============================================================================
def add(a, b):
    """The UNIT: one binary add (2 reads + 1 write). The baseline is 2x this."""
    return a + b


def add_indep2(a, b, c, d):
    """NO-DEPENDENCY control: two INDEPENDENT adds. Same 4R:2W bytes as add3, but neither add
    reads the other's output -> isolates the pure byte count from the dependency."""
    return a + b, c + d


def add3_fused(a, b, c):
    """add3 FUSED: the dependent chain ((a+b)+c) in ONE compiled graph -> a single kernel."""
    return (a + b) + c


def make_add3_separate():
    """add3 as SEPARATE kernels. Each `+` is its OWN compiled kernel (this outer function is
    NOT torch.compiled), so the intermediate (a+b) is written to HBM by kernel 1 and read back
    by kernel 2 -- the same read-after-write dependency as the fused add3, but no fusion."""
    compiled_add = torch.compile(add)  # one compiled binary add, launched twice

    def add3_sep(a, b, c):
        return compiled_add(compiled_add(a, b), c)

    return add3_sep


# ============================================================================
# HOW EACH IS RUN: compile under the chosen scratchpad setting, warm it, then take
# ONE profiled trace whose FULL table we print, plus the median kernel time.
# ============================================================================
def sync(out):  # move result(s) to host so the device actually finishes the work
    for t in out if isinstance(out, (tuple, list)) else (out,):
        t.cpu()


def kernel_us(prof):
    """Sum the compute-event device time in a trace (Memset/Memcpy excluded -- the same
    classification profile_ops.py uses). For a separate-kernel chain this sums all sub-kernels."""
    total = 0.0
    for ev in prof.key_averages():
        us = getattr(ev, "self_device_time_total", 0) or 0
        name = ev.key or ""
        if us > 0 and "Memset" not in name and "Memcpy" not in name:
            total += us
    return total


def profile_op(name, build, n_inputs, lx):
    """Compile the op under scratchpad planning = `lx`, warm it, run REPS profiled traces, PRINT
    the full profiler table for the first trace, and return the median kernel time (us)."""
    spyre_config.lx_planning = lx  # read at compile time in torch_spyre/_inductor/passes.py
    torch._dynamo.reset()  # force a fresh compile under this lx setting
    fn = build()
    args = inputs(n_inputs)
    for _ in range(WARMUP):
        sync(fn(*args))
    reps, first = [], None
    for i in range(REPS):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            sync(fn(*args))
        if i == 0:
            first = prof
        k = kernel_us(prof)
        if k > 0:
            reps.append(k)
    med = statistics.median(reps) if reps else float("nan")

    print("\n" + "=" * 78)
    print(f"OP: {name}   (lx_planning={lx}, ROWS={ROWS}, COLS={COLS})")
    print("=" * 78)
    print(first.key_averages().table(sort_by="cuda_time_total", row_limit=100).replace("CUDA", "AIU"))
    print(f"--> compute kernel time (median over {REPS} reps, Memset/Memcpy excluded): {med:.1f} us")
    return med


def main():
    results = [
        ("add (single)", profile_op("add (single)", lambda: torch.compile(add), 2, False)),
        ("add_indep2 (no dep)", profile_op("add_indep2 (no dep)", lambda: torch.compile(add_indep2), 4, False)),
        ("add3_sep (dep, separate)", profile_op("add3_sep (dep, separate)", make_add3_separate, 3, False)),
        ("add3 (dep, fused)", profile_op("add3 (dep, fused)", lambda: torch.compile(add3_fused), 3, False)),
        ("add3, LX on (buf on-chip)", profile_op("add3, LX on (buf on-chip)", lambda: torch.compile(add3_fused), 3, True)),
    ]

    single = results[0][1]
    baseline = 2 * single  # byte count of a 2-add chain (fig3's dashed baseline)
    print("\n" + "#" * 78)
    print(f"# SUMMARY -- kernel time per op  (ROWS={ROWS}, COLS={COLS}, reps={REPS})")
    print(f"# byte-count baseline = 2 x add(single) = 2 x {single:.0f} = {baseline:.0f} us")
    print("#" * 78)
    print(f"{'op':>28} | {'kernel us':>10} | {'vs 2x add baseline':>18}")
    print("-" * 64)
    for name, us in results:
        vs = "-- (this is 1x add)" if name.startswith("add (single)") else f"{100 * (us / baseline - 1):>+16.0f}%"
        print(f"{name:>28} | {us:>10.1f} | {vs:>18}")
    print(
        "\nExpected: add_indep2 ~0%, add3_sep ~+7%, add3 ~+7% (== separate -> the dependency is\n"
        "the margin, fusion is free), add3 LX-on ~-34% (the HBM round-trip is gone)."
    )


if __name__ == "__main__":
    main()
