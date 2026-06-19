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

"""Cross-check our SPYRE_PROFILE_SYNC min-of-N against the PyTorch profiler.

The torch.profiler ``PrivateUse1`` activity (needs the kineto-spyre wheel -- see
docs/source/user_guide/profiling/pytorch_profiler.md) reports a **"Self SPYRE"**
column = the TRUE per-kernel device time. Two uses:

1. **Validate our timer.** Our SPYRE_PROFILE_SYNC min brackets the launch+sync, so it
   = device time + ~7 us host residue. Run the SAME op/size here and in bench_ops.py:
   ``our_min  ~=  Self_SPYRE(sdsc_fused_*)  +  ~7us`` confirms the measurement.
2. **Decompose the ~20 us fixed term.** The profiler surfaces a separate
   ``Memset (Device)`` event (~12.5 us on a tiny add). If it stays ~constant across
   sizes, a big chunk of our "fixed" is a real DEVICE memset, not host overhead.

NOTE: this is a TIME + memory-allocation profiler -- it has NO DRAM bandwidth / read-
vs-write / bus-utilization counters, so it CANNOT explain why read+write halves
bandwidth (that needs aiu-smi). It only gives cleaner device time.

Knobs: BENCH_OP, BENCH_ROWS, BENCH_COLS, BENCH_WARMUP. BENCH_OP is any of the
bench_ops/bench_bandwidth ops: neg copy gelu relu sigmoid exp | mul add | add3 add4 |
read sumrow sumall amax mean | bcast mulbcast | write.

Examples:
    # one op (prints the table + a parseable SUMMARY line with kernel_us / memset_us)
    BENCH_OP=neg BENCH_COLS=1024 \
        python docs/source/user_guide/examples/profile_ops.py
    # full golden re-sweep (rebuild the model from kernel time):
    bash docs/source/user_guide/examples/run_profile_sweep.sh
"""

import os

# Enable the cost-model dump so we read ITS device-layout I/O size (the same byte
# accounting the model uses), and force compile so the dump fires.
os.environ.setdefault("SPYRE_DUMP_COST", "1")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from torch_spyre._inductor import cost_model, dump_cost_model  # noqa: E402

DEVICE = torch.device("spyre")
OP = os.environ.get("BENCH_OP", "gelu")
ROWS = int(os.environ.get("BENCH_ROWS", "512"))
COLS = int(os.environ.get("BENCH_COLS", "16384"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
SENCORES = os.environ.get("SENCORES", "")  # core count (read only to tag the SUMMARY)
TILES = int(os.environ.get("BENCH_TILES", "0"))  # coarse-tile dim0 into K (>=2 on)
LX = os.environ.get("LX_PLANNING", "1")  # scratchpad planning on(1)/off(0); SUMMARY tag

torch.manual_seed(0xAFFE)


def _rand(*shape):
    return torch.rand(*shape, dtype=torch.float16).to(DEVICE)


def _sum_all(*ts):
    acc = ts[0]
    for t in ts[1:]:
        acc = acc + t
    return acc


# Same ops as bench_ops/bench_bandwidth so the GOLDEN kernel times map 1:1.
_UNARY = {  # 1 read + 1 write (gelu/relu/sigmoid/exp also probe arithmetic-free)
    "neg": lambda x: -x,  # cleanest balanced 1R+1W (no constant)
    "copy": lambda x: x + 1.0,  # scalar 1.0 is a cached broadcast -> still 1R+1W
    "gelu": F.gelu,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "exp": torch.exp,
}
_BINARY = {"mul": lambda a, b: a * b, "add": lambda a, b: a + b}  # 2R + 1W
_NARY = {"add3": 3, "add4": 4}  # n inputs summed (intermediates staged in LX)
_REDUCE = {  # read-dominated; sumall reduces to a scalar -> ring combine
    "read": lambda x: x.sum(dim=-1),
    "sumrow": lambda x: x.sum(dim=-1),  # reduce COLS -> [ROWS] (within-stick axis)
    "sumcol": lambda x: x.sum(dim=0),  # reduce ROWS -> [COLS] (the other axis)
    "sumall": lambda x: x.sum(),  # -> scalar: reduced axis split across all cores
    "amax": lambda x: x.amax(dim=-1),
    "mean": lambda x: x.mean(dim=-1),
}
_BCAST = {"bcast": lambda a, b: a + b, "mulbcast": lambda a, b: a * b}  # b = [1, COLS]
# Coarse-tiled dim0 reductions (mirror coarse_tile/run_{sum,amax,amin}_dim0_tiled.py):
# reduce a [B=ROWS, D=COLS] tensor over B, tiling B into BENCH_TILES chunks. With
# BENCH_TILES>=2 a spyre_hint wraps it in a K-iteration loop (fill + K x reduce/combine)
# BENCH_TILES<=1 is the plain untiled baseline. Run each under LX_PLANNING=0/1.
_CT_REDUCE = {"ctsum": "sum", "ctamax": "amax", "ctamin": "amin"}


# Per-call setup hook: the coarse-tile path sets this to re-declare its named dims
# before every traced invocation (the example declares once + compiles once; this
# profiling harness traces repeatedly). None for all non-tiled ops.
_PREPARE = None


def _ct_workload(rtype: str):
    """Coarse-tiled dim0 reduction, mirroring coarse_tile/run_*_dim0_tiled.py + utils.py
    ``_compile_and_run``: a CPU input (sum scaled 0.1 as the example does), eager
    ``declare_tensor_dim``, ``name_tensor_dims`` + ``spyre_hint`` inside fn, an eager
    reference call of fn, then a dynamo/Fx cache reset right before compile.
    """
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    b, d = ROWS, COLS
    scale = 0.1 if rtype == "sum" else 1.0  # example scales sum to avoid fp16 growth
    x_cpu = torch.randn(b, d, dtype=torch.float16) * scale

    def _declare():
        pnd.declare_tensor_dim("B", b)
        pnd.declare_tensor_dim("D", d)

    _declare()  # eager, as the example does in main() before compiling

    def reduce_fn(x):
        return getattr(x, rtype)(dim=0)

    if TILES >= 2:

        def fn(x):
            pnd.name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"B": TILES}):
                return reduce_fn(x)

        # Re-declare EAGERLY (not inside fn -> no trace perturbation) before each traced
        # call so propagate_named_dims always resolves the named-dim sizes.
        _PREPARE = _declare
    else:
        fn = reduce_fn

    fn(x_cpu)  # eager reference call (mirror compare_with_cpu's cpu_result = fn(...))
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (x_cpu.to(DEVICE),)


def make_workload():
    if OP in _UNARY:
        return torch.compile(_UNARY[OP]), (_rand(ROWS, COLS),)
    if OP in _BINARY:
        return torch.compile(_BINARY[OP]), (_rand(ROWS, COLS), _rand(ROWS, COLS))
    if OP in _NARY:
        xs = tuple(_rand(ROWS, COLS) for _ in range(_NARY[OP]))
        return torch.compile(_sum_all), xs
    if OP in _REDUCE:
        return torch.compile(_REDUCE[OP]), (_rand(ROWS, COLS),)
    if OP in _BCAST:  # row-vector broadcast: a[R,C] + b[1,C] (b cached across rows)
        return torch.compile(_BCAST[OP]), (_rand(ROWS, COLS), _rand(1, COLS))
    if OP == "bcastcol":  # col-vector broadcast: a[R,C] + b[R,1] (b cached across cols)
        return torch.compile(lambda a, b: a + b), (_rand(ROWS, COLS), _rand(ROWS, 1))
    if OP == "write":  # write-only: both inputs broadcast -> cached
        return torch.compile(lambda b, c: b + c), (_rand(1, COLS), _rand(ROWS, 1))
    if OP in _CT_REDUCE:  # coarse-tiled dim0 reduction (BENCH_TILES, LX_PLANNING)
        return _ct_workload(_CT_REDUCE[OP])
    known = (
        list(_UNARY) + list(_BINARY) + list(_NARY) + list(_REDUCE) + list(_BCAST)
        + ["bcastcol", "write"] + list(_CT_REDUCE)
    )
    raise SystemExit(f"unknown BENCH_OP={OP!r} (use {known})")


def _print_io(io: dict) -> None:
    """Per-tensor device-layout I/O the COST MODEL counts: dims, residency, bytes.

    Lines are prefixed ``IO `` so the sweep (run_profile_sweep.sh) can grep them
    alongside the SUMMARY line instead of dropping the breakdown.
    """
    print("IO -- device-layout I/O (cost model, stick-padded) --")
    for o in io.get("ops", []):
        red = " [reduction]" if o.get("is_reduction") else ""
        print(f"IO   op {o['name']}{red}")
        for a in o["args"]:
            bc = " broadcast (loaded once)" if a["broadcast"] else ""
            lf = a.get("loop_factor", 1)
            xl = f" xL={lf}" if lf > 1 else ""
            log = f"torch {a['logical']} -> " if a.get("logical") else ""
            print(
                f"IO     {a['role']:<6} {a.get('name', '?'):<22} "
                f"{log}device {a['dims']} in {a['mem'].upper()} = "
                f"{a['elems']} elems x 2B = {a['bytes']} B"
                f"  (hbm counted: {a['hbm_counted']} B){xl}{bc}"
            )
    print(f"IO   => HBM I/O total = {io.get('hbm_bytes', 0)} B  "
          f"(lx {io.get('lx_bytes', 0)} B, ~free)")


def _print_model(feats: list) -> float:
    """Print the cost model's ESTIMATED kernel time + rough calc, after the I/O dump.

    Lines are prefixed ``MODEL `` so the sweep greps them. Returns the predicted us so
    the SUMMARY can carry it next to the measured kernel_us.
    """
    if not feats:
        print("MODEL (no features extracted)")
        return 0.0
    p = cost_model.CostParams()
    r = sum(o.read_bytes() for o in feats)
    w = sum(o.write_bytes() for o in feats)
    lp = max((o.loop_trip for o in feats), default=1)
    base = (r + w) / p.bw_peak_gbps
    turn = p.rw_turnaround_ns_per_byte * min(r, w)
    t = cost_model.predict_ops(feats, p)
    print(f"MODEL -- estimate (turnaround): T = (R+W)/BW_PEAK + a*min(R,W)"
          f"{' + c_loop*L' if lp > 1 else ''} --")
    print(f"MODEL   R={r} B (read)   W={w} B (write)   loop_trip L={lp}")
    print(f"MODEL   base = (R+W)/BW_PEAK = ({r}+{w})/{p.bw_peak_gbps:.0f} "
          f"= {base / 1000:.2f} us")
    print(f"MODEL   turn = a*min(R,W) = {p.rw_turnaround_ns_per_byte}*{min(r, w)} "
          f"= {turn / 1000:.2f} us")
    if lp > 1:
        loop = p.c_loop_ns * lp
        print(f"MODEL   loop = c_loop*L = {p.c_loop_ns}*{lp} = {loop / 1000:.2f} us")
    print(f"MODEL   => T_model = {t / 1000:.2f} us")
    return t / 1000


def _run():
    compiled, args = make_workload()
    for _ in range(WARMUP):  # compile (-> cost-model dump fires) + warm the kernel
        if _PREPARE is not None:  # coarse-tile: re-declare named dims before each trace
            _PREPARE()
        compiled(*args).cpu()
    io = dict(dump_cost_model.LAST_IO)  # device-layout I/O the model computed
    feats = list(dump_cost_model.LAST_FEATS)  # raw OpFeatures for predict_ops()
    io_hbm_bytes = io.get("hbm_bytes", 0)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        if _PREPARE is not None:  # coarse-tile: re-declare before the profiled trace
            _PREPARE()
        compiled(*args).cpu()

    print(f"== {OP}[{ROWS}x{COLS}] -- profiler kernel time vs cost-model I/O ==")
    ka = prof.key_averages()
    print(ka.table(sort_by="cuda_time_total", row_limit=20).replace("CUDA", "AIU"))
    _print_io(io)
    pred_us = _print_model(feats)  # cost-model estimate + rough calc (after I/O dump)

    # Parseable one-liner for a size sweep -> re-fit fill + BW on the GOLDEN kernel
    # time, and track the (non-deterministic) Memset overhead separately.
    kernel = memset = other = 0.0
    for ev in ka:
        us = getattr(ev, "self_device_time_total", 0) or getattr(
            ev, "self_cuda_time_total", 0
        )
        if not us or us <= 0:
            continue
        if "sdsc_fused" in ev.key:
            kernel += us
        elif "Memset" in ev.key:
            memset += us
        else:
            other += us
    # Effective BW from the GOLDEN kernel time and the model's device-layout I/O.
    bw = io_hbm_bytes / (kernel * 1000) if kernel > 0 else 0.0
    err = (pred_us - kernel) / kernel * 100.0 if kernel > 0 else 0.0
    print(
        f"SUMMARY op={OP} rows={ROWS} cols={COLS} cores={SENCORES or '-'} "
        f"tiles={TILES} lx={LX} io_hbm_bytes={io_hbm_bytes} "
        f"kernel_us={kernel:.3f} pred_us={pred_us:.3f} err_pct={err:+.1f} "
        f"bw_gbps={bw:.1f} memset_us={memset:.3f} "
        f"other_dev_us={other:.3f} total_dev_us={kernel + memset + other:.3f}"
    )


def main():
    # Self-report failures: print the full traceback (stderr) AND a parseable FAILED
    # SUMMARY (stdout) carrying the reason, so a sweep that greps SUMMARY still records
    # WHY a run died instead of a bare "FAILED".
    try:
        _run()
    except Exception as exc:  # noqa: BLE001 - diagnostic wrapper
        import traceback

        traceback.print_exc()
        print(
            f"SUMMARY op={OP} rows={ROWS} cols={COLS} tiles={TILES} lx={LX} "
            f"FAILED reason={type(exc).__name__}: {str(exc)[:140]}"
        )


if __name__ == "__main__":
    main()
