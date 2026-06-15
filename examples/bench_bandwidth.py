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

"""Pure-I/O DRAM bandwidth probe for Spyre (Exp B in the cost-model plan).

Why: our pointwise cost model fits an effective DRAM rate of ~111 GB/s, but the
compiler's own matmul model uses the LPDDR5 *peak* of 204.8 GB/s
(``_HBM_BW_GBS`` in ``torch_spyre/_inductor/work_division.py``). This probe pushes
large, low-compute memory traffic across all cores to find what bandwidth is
actually achievable and explain the ~1.85x gap (204.8 / 111).

Two workloads, contrasted:
    copy : y = x + 1.0        2-stream  (1 read + 1 write)  -> read+write BW
    read : y = x.sum(dim=-1)  read-only (~1 read, tiny write) -> isolates read BW

If ``read`` (1 pass) is ~half the latency of ``copy`` (2 passes) at the same size,
read and write share one rate (no bidirectional penalty). If ``read`` is faster
than that, reads are cheaper than writes -> the cap is read+write contention.

The effective BW here is bytes / device-min-latency (== GB/s). The fixed ~20 us
per-kernel term depresses it at small sizes, so SWEEP size: the asymptote (large
size, or the slope d(bytes)/d(latency) across the sweep) is the real ceiling.

Pair with ``aiu-smi`` for the ACTUAL DDR bandwidth (latency-independent): set
``BENCH_BW_SUSTAIN_S=20`` to keep the device saturated for a sampling window, and
run ``aiu-smi`` in a second shell (see docs/source/user_guide/profiling/
device_monitoring.md). If aiu-smi reads ~200 while our latency BW reads ~111, the
gap is host/ramp overhead; if aiu-smi also reads ~111, that IS the ceiling.

Knobs:
    BENCH_BW_OP        copy | read              (default copy)
    BENCH_ROWS, BENCH_COLS                      shape (default 512 x 8192)
    BENCH_RUNS, BENCH_WARMUP
    BENCH_BW_SUSTAIN_S N   after measuring, loop the op for ~N seconds so aiu-smi
                           can sample DDR bandwidth (default 0 = off)
    SENCORES           cores (runtime env; default 32)

Examples:
    # size sweep: does effective BW plateau at ~111 or keep climbing toward 204.8?
    for n in 1024 4096 16384 65536; do \
      BENCH_BW_OP=copy BENCH_COLS=$n python examples/bench_bandwidth.py; done
    # read-only vs read+write at one big size
    BENCH_BW_OP=read BENCH_COLS=16384 python examples/bench_bandwidth.py
    # aiu-smi window (start aiu-smi in another shell during the sustained phase)
    BENCH_BW_OP=copy BENCH_COLS=65536 BENCH_BW_SUSTAIN_S=20 \
      python examples/bench_bandwidth.py
"""

import os
import time

os.environ.setdefault("SPYRE_PROFILE", "1")
os.environ.setdefault("SPYRE_PROFILE_SYNC", "1")

import torch  # noqa: E402

from torch_spyre.execution import profiling  # noqa: E402
from torch_spyre.execution.bench import measure_device  # noqa: E402

DEVICE = torch.device("spyre")
OP = os.environ.get("BENCH_BW_OP", "copy")
ROWS = int(os.environ.get("BENCH_ROWS", "512"))
COLS = int(os.environ.get("BENCH_COLS", "8192"))
RUNS = int(os.environ.get("BENCH_RUNS", "100"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "20"))
SUSTAIN_S = float(os.environ.get("BENCH_BW_SUSTAIN_S", "0"))

DT = 2  # fp16 bytes
FILL_NS = 20_000.0  # calibrated per-kernel fixed term (cost_model.py)
PEAK_GBPS = 204.8  # _HBM_BW_GBS: LPDDR5 aggregate peak (work_division.py)

# op -> (fn, hbm_passes). passes = memory passes counted for the BW estimate:
#   copy = read x + write y              -> 2
#   read = read x + write tiny reduction -> ~1 (read-only; the [ROWS] write is
#          negligible next to the [ROWS,COLS] read)
_OPS = {
    "copy": (lambda x: x + 1.0, 2),
    "read": (lambda x: x.sum(dim=-1), 1),
}

torch.manual_seed(0xAFFE)


def _sync():
    """Drain the device (no D2H copy); no-op if the runtime sync is absent."""
    try:
        import torch_spyre._C as _C

        _C.synchronize()
    except Exception:  # noqa: BLE001 - tolerate a runtime without sync
        pass


def main():
    if OP not in _OPS:
        raise SystemExit(f"unknown BENCH_BW_OP={OP!r} (use {list(_OPS)})")
    fn, passes = _OPS[OP]
    x = torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE)
    compiled = torch.compile(fn)

    elems = ROWS * COLS
    bytes_moved = passes * elems * DT
    cores = os.environ.get("SENCORES", "default(32)")
    profiling.set_report_at_exit(False)
    print(
        f"bandwidth probe: op={OP} shape=[{ROWS}x{COLS}] elems={elems} "
        f"passes={passes} bytes={bytes_moved}  SENCORES={cores}"
    )

    snap = measure_device(lambda: compiled(x), runs=RUNS, warmup=WARMUP)
    print(profiling.format_report())

    # Total device time for one workload = sum of its kernels' deterministic mins
    # (kernels run sequentially within the call; each min is jitter-stripped).
    n_kernels = len(snap)
    total_min_ns = sum(rec["min_ns"] for rec in snap.values()) if snap else 0.0
    if total_min_ns > 0:
        bw = bytes_moved / total_min_ns  # bytes/ns == GB/s
        net_ns = total_min_ns - FILL_NS * n_kernels  # drop per-kernel fixed term
        line = (
            f"  effective BW: {bw:6.1f} GB/s  "
            f"({bytes_moved} B / {total_min_ns / 1000:.2f} us, "
            f"{n_kernels} kernel(s))  = {bw / PEAK_GBPS * 100:.0f}% of {PEAK_GBPS} peak"
        )
        if net_ns > 0:
            bw_net = bytes_moved / net_ns
            line += (
                f"\n  BW excl. fill: {bw_net:6.1f} GB/s "
                f"(minus {FILL_NS / 1000:.0f}us x {n_kernels}; "
                f"true asymptote = sweep slope)"
            )
        print(line)

    if SUSTAIN_S > 0:
        # Keep the device saturated so aiu-smi can read steady-state DDR BW.
        # Back-to-back async launches with a periodic sync bound the queue.
        print(
            f"=== SUSTAINED I/O START: start aiu-smi NOW; "
            f"saturating ~{SUSTAIN_S:.0f}s ===",
            flush=True,
        )
        t0 = time.perf_counter()
        launches = 0
        while time.perf_counter() - t0 < SUSTAIN_S:
            for _ in range(50):
                compiled(x)
            launches += 50
            _sync()
        wall = time.perf_counter() - t0
        host_bw = launches * bytes_moved / (wall * 1e9)
        print(
            f"=== SUSTAINED I/O END: {launches} launches in {wall:.1f}s, "
            f"host-side BW {host_bw:.1f} GB/s (compare to aiu-smi DDR reading) ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
