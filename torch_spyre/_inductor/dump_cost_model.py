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

"""Extract cost-model features from the after-pre-scheduling LoopLevel IR.

Walks ``graph.operations`` and builds :class:`cost_model.OpFeatures` per op
(per-core cores, per-tensor-arg bytes + HBM/LX residency + broadcast flags),
then a dump hook (``SPYRE_DUMP_COST=1``) prints the features and the predicted
device latency so it can be compared against the measured value on hardware.

Extraction is best-effort and defensive: anything it can't resolve falls back to
a safe default and never raises into compilation. The numbers must be validated
against device measurements (``examples/bench_*``); the model is only as good as
this extraction.
"""

import os

from torch._inductor.ir import ComputedBuffer

from .cost_model import ArgTraffic, OpFeatures, explain
from .pass_utils import apply_splits_from_index_coeff, iteration_space_from_op


def cost_dump_enabled() -> bool:
    return os.environ.get("SPYRE_DUMP_COST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _int(x, default: int = 1) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _prod_ints(seq) -> int:
    n = 1
    for s in seq:
        n *= _int(s, 1)
    return n


def _op_name(op) -> str:
    data = getattr(op, "data", None)
    node = getattr(data, "origin_node", None)
    if node is not None:
        return getattr(node, "name", None) or str(getattr(node, "target", node))
    rtype = getattr(data, "reduction_type", None)
    if rtype:
        return str(rtype)
    return type(data).__name__ if data is not None else op.get_operation_name()


def _cores(op) -> int:
    splits = getattr(op, "op_it_space_splits", None)
    if not splits:
        return 1
    try:
        rw = op.get_read_writes()
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        it_space = iteration_space_from_op(op)
        readable = apply_splits_from_index_coeff(
            splits, write_index, read_index, it_space
        )
        return _prod_ints(readable.values())
    except Exception:  # noqa: BLE001 - best-effort feature extraction
        return 1


def _mem_of_layout(layout) -> str:
    alloc = getattr(layout, "allocation", None)
    if isinstance(alloc, dict) and "lx" in alloc:
        return "lx"
    return "hbm"


def _device_dims(layout):
    """Stick-padded DEVICE dims (e.g. [4, 512, 64]) from a committed FixedTiledLayout
    -- the TRUE shape that moves (sticks are 64 fp16 elems; a row of N rounds up to
    ceil(N/64)*64). None when the device layout isn't available (use logical instead).
    """
    dl = getattr(layout, "device_layout", None)
    ds = getattr(dl, "device_size", None) if dl is not None else None
    if not ds:
        return None
    try:
        return [_int(x, 1) for x in ds]
    except Exception:  # noqa: BLE001 - symbolic/unresolved
        return None


def _input_traffic(name: str):
    """(mem, dims, elems) for a read buffer -- ``dims`` is the device (stick) shape,
    ``elems`` its product (logical fallback if the device layout isn't committed). So a
    reduction's reduced input is naturally full-sized, with no reduction_size scaling.
    Returns (None, None, None) if the buffer can't be resolved (caller falls back)."""
    try:
        from torch._inductor.virtualized import V

        buf = V.graph.get_buffer(name)
        if buf is not None:
            layout = buf.get_layout()
            dims = _device_dims(layout)
            logical = list(buf.get_size())
            elems = _prod_ints(dims) if dims else _prod_ints(logical)
            return _mem_of_layout(layout), (dims if dims else logical), elems
    except Exception:  # noqa: BLE001 - graph inputs / unresolved
        pass
    return None, None, None


def _loop_features(op):
    """(loop_trip, tiles_reduction_dim) from the coarse-tiling ``loop_info`` on the op
    (loop_info.py / coarse_tile.py). ``loop_trip`` = product of ``loop_count`` (1 if not
    tiled). ``tiles_reduction_dim`` = ``loop_tiled_reduction_dims`` is non-empty. NOTE
    the fill/combine ops carry the same ``loop_info`` (so they also report a tiled
    reduction dim) -- only the REDUCTION op's input actually advances, so the caller
    ANDs this with ``is_reduction``. SCOPE: reduction-dim tiling (sum/amax/amin); output
    (pointwise) tiling would need per-arg advancing detection."""
    li = getattr(op, "loop_info", None)
    if li is None:
        return 1, False
    trip = 1
    for c in getattr(li, "loop_count", None) or []:
        trip *= _int(c, 1)
    red_dims = getattr(li, "loop_tiled_reduction_dims", None) or []
    return max(1, trip), any(bool(level) for level in red_dims)


def extract_op_features(op) -> OpFeatures:
    """Build OpFeatures for one ComputedBuffer op (best-effort)."""
    data = getattr(op, "data", None)
    is_reduction = getattr(data, "reduction_type", None) is not None
    loop_trip, tiles_red_dim = _loop_features(op)
    # Only a REDUCTION op that tiles a reduction dim has an advancing (read-once) input;
    # the fill/combine ops share the loop_info but their accumulators are re-read each
    # iteration (factor L). So AND the loop flag with is_reduction.
    is_tiled_red = is_reduction and tiles_red_dim
    dtype_bytes = _int(getattr(op.get_dtype(), "itemsize", 2), 2)
    out_size = list(op.get_size())
    # TRUE I/O sizes come from the committed DEVICE layout (sticks), not the torch
    # logical shape -- a row of N fp16 rounds up to ceil(N/64)*64, and reduction/
    # broadcast operands carry their own device size.
    out_dims = _device_dims(op.get_layout()) or out_size
    out_elems = _prod_ints(out_dims)

    cores = _cores(op)

    # Cross-core ring combine: work division splits OUTPUT dims first, then the reduced
    # axis with leftover cores -> the reduced axis is split only when out_elems < cores.
    # Approx k as the cores not absorbed by the output (refine if rung 11 needs it).
    reduction_cores = 1
    if is_reduction and out_elems < cores:
        reduction_cores = max(1, cores // max(1, out_elems))

    args: list = []
    # Output arg (device-sized). In a coarse-tiling loop the output (a per-tile partial
    # or an accumulator) is re-written every iteration at a fixed address -> factor = L.
    args.append(
        ArgTraffic(
            name=op.get_operation_name(),
            role="output",
            mem=_mem_of_layout(op.get_layout()),
            elems=out_elems,
            dims=list(out_dims),
            loop_factor=loop_trip,
        )
    )
    # Input args, from the op's reads. Each read is sized by ITS OWN buffer's device
    # layout -- so a reduction's reduced input is naturally full-sized (no separate
    # reduction scaling), and a broadcast operand carries its real (one-row) size.
    try:
        reads = op.get_read_writes().reads
    except Exception:  # noqa: BLE001
        reads = []
    n_out_vars = len(out_size)
    for dep in reads:
        name = getattr(dep, "name", "?")
        index = getattr(dep, "index", None)
        # Broadcast heuristic: the read index references fewer loop variables than
        # the output rank -> it is loaded ONCE and reused across the broadcast dim, so
        # it is counted at its own (small) device size, not the output size. This
        # INCLUDES scalars/constants (0 loop vars, e.g. the `1.0` in `x + 1.0`): a
        # scalar is the maximally-broadcast input -- its one-load size is ~1 stick, so
        # it costs ~nothing, but it is no longer forced to exactly zero.
        broadcast = False
        try:
            n_index_vars = len(getattr(index, "free_symbols", []) or [])
            broadcast = n_index_vars < n_out_vars
        except Exception:  # noqa: BLE001
            broadcast = False
        mem, dims, in_elems = _input_traffic(name)
        if in_elems is None:  # unresolved buffer -> fallback
            # A broadcast operand with no resolvable buffer (e.g. a scalar constant)
            # is loaded once and is at most ~1 element -- do NOT inflate it to the
            # output size. Only a NON-broadcast unresolved read is conservatively
            # sized at the full output.
            if broadcast:
                mem, dims, in_elems = "hbm", [1], 1
            else:
                mem, dims, in_elems = "hbm", list(out_dims), out_elems
        # Loop scaling: the reduced input of a tiled REDUCTION advances (walks the full
        # tensor once across tiles) -> factor 1, its full device_size already covers all
        # tiles. Every other looped input (a combine's accumulator / partial, re-read
        # each iteration at a fixed address) -> factor L. Non-tiled ops: L = 1.
        in_loop_factor = 1 if is_tiled_red else loop_trip
        args.append(
            ArgTraffic(
                name=name,
                role="input",
                mem=mem,
                elems=in_elems,
                broadcast=broadcast,
                dims=list(dims),
                loop_factor=in_loop_factor,
            )
        )

    return OpFeatures(
        name=_op_name(op),
        is_reduction=is_reduction,
        out_elems=out_elems,
        cores=cores,
        dtype_bytes=dtype_bytes,
        args=args,
        reduction_cores=reduction_cores,
        loop_trip=loop_trip,
    )


def extract_features(operations: list) -> list:
    """Build OpFeatures for every ComputedBuffer op in the graph."""
    feats = []
    for op in operations:
        if isinstance(op, ComputedBuffer):
            try:
                feats.append(extract_op_features(op))
            except Exception:  # noqa: BLE001 - skip ops we can't model
                continue
    return feats


# Totals + per-arg detail from the most recent extraction, using the DEVICE-layout
# byte accounting. Tools (e.g. examples/profile_ops.py) read this to get the model's
# I/O size and verify BW = hbm_bytes / kernel_time, without re-parsing the printed dump.
LAST_IO: dict = {}


def _record_last_io(feats: list) -> None:
    global LAST_IO
    ops = []
    for o in feats:
        args = []
        for a in o.args:
            bs = a.elems * o.dtype_bytes
            # Every HBM arg counts at its own size x loop_factor (L for a per-tile
            # accumulator re-accessed each loop iteration, 1 otherwise); broadcast
            # operands carry their small one-load size (counted, not zeroed). LX ~free.
            counted = bs * a.loop_factor if a.mem == "hbm" else 0
            args.append(
                {
                    "role": a.role,
                    "mem": a.mem,
                    "dims": list(a.dims) if a.dims else [a.elems],
                    "elems": a.elems,
                    "loop_factor": a.loop_factor,
                    "bytes": bs,
                    "hbm_counted": counted,
                    "broadcast": a.broadcast,
                }
            )
        ops.append({"name": o.name, "is_reduction": o.is_reduction, "args": args})
    LAST_IO = {
        "hbm_bytes": sum(o.hbm_bytes() for o in feats),
        "lx_bytes": sum(o.lx_bytes() for o in feats),
        "ops": ops,
    }


def dump_cost_model(operations: list) -> None:
    """Print per-op cost features + predicted latency; no-op unless SPYRE_DUMP_COST.

    Treats the whole op list as one bundle (matching full fusion, e.g. softmax);
    for single-op example programs this is just that op.
    """
    if not cost_dump_enabled():
        return
    from .dump_common import banner, emit

    try:
        feats = extract_features(operations)
        _record_last_io(feats)
        bar = banner("Cost model features + prediction (after pre-scheduling)")
        emit(f"{bar}\n{explain(feats)}\n")
    except Exception as exc:  # noqa: BLE001 - instrumentation must not raise
        emit(f"[SPYRE_DUMP_COST] failed: {exc!r}")
