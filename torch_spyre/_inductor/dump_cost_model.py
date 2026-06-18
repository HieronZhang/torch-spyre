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


def _device_elems(layout, fallback: int) -> int:
    """Stick-padded DEVICE element count from a committed FixedTiledLayout -- the TRUE
    I/O size (sticks are 64 fp16 elems; a row of N rounds up to ceil(N/64)*64). Falls
    back to the logical count when the device layout isn't available. Use this, not the
    torch logical shape, so the byte count reflects what actually moves on device.
    """
    dl = getattr(layout, "device_layout", None)
    ds = getattr(dl, "device_size", None) if dl is not None else None
    try:
        if ds:
            return _prod_ints(ds)
    except Exception:  # noqa: BLE001 - symbolic/unresolved -> logical fallback
        pass
    return fallback


def _input_traffic(name: str):
    """(mem, device_elems) for a read buffer. ``device_elems`` is the buffer's own
    stick-padded device size (logical size if uncommitted) -- so a reduction's reduced
    input is naturally full-sized, with no separate reduction_size scaling. Returns
    (None, None) if the buffer can't be resolved (caller falls back)."""
    try:
        from torch._inductor.virtualized import V

        buf = V.graph.get_buffer(name)
        if buf is not None:
            layout = buf.get_layout()
            elems = _device_elems(layout, _prod_ints(buf.get_size()))
            return _mem_of_layout(layout), elems
    except Exception:  # noqa: BLE001 - graph inputs / unresolved
        pass
    return None, None


def extract_op_features(op) -> OpFeatures:
    """Build OpFeatures for one ComputedBuffer op (best-effort)."""
    data = getattr(op, "data", None)
    is_reduction = getattr(data, "reduction_type", None) is not None
    dtype_bytes = _int(getattr(op.get_dtype(), "itemsize", 2), 2)
    out_size = list(op.get_size())
    # TRUE I/O sizes come from the committed DEVICE layout (sticks), not the torch
    # logical shape -- a row of N fp16 rounds up to ceil(N/64)*64, and reduction/
    # broadcast operands carry their own device size.
    out_elems = _device_elems(op.get_layout(), _prod_ints(out_size))

    cores = _cores(op)

    # Cross-core ring combine: work division splits OUTPUT dims first, then the reduced
    # axis with leftover cores -> the reduced axis is split only when out_elems < cores.
    # Approx k as the cores not absorbed by the output (refine if rung 11 needs it).
    reduction_cores = 1
    if is_reduction and out_elems < cores:
        reduction_cores = max(1, cores // max(1, out_elems))

    args: list = []
    # Output arg (device-sized).
    args.append(
        ArgTraffic(
            name=op.get_operation_name(),
            role="output",
            mem=_mem_of_layout(op.get_layout()),
            elems=out_elems,
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
        # Broadcast heuristic: the read index references fewer loop variables
        # than the output rank -> it is loaded once and cached (~free), like the
        # rung-6 broadcasts. This INCLUDES scalars/constants (0 loop vars, e.g. the
        # `1.0` in `x + 1.0`): a scalar is the maximally-broadcast input, not a full
        # HBM read. (The old `0 < n_index_vars` wrongly counted scalars as full.)
        broadcast = False
        try:
            n_index_vars = len(getattr(index, "free_symbols", []) or [])
            broadcast = n_index_vars < n_out_vars
        except Exception:  # noqa: BLE001
            broadcast = False
        mem, in_elems = _input_traffic(name)
        if in_elems is None:  # unresolved buffer -> conservative fallback
            mem, in_elems = "hbm", out_elems
        args.append(
            ArgTraffic(
                name=name,
                role="input",
                mem=mem,
                elems=in_elems,
                broadcast=broadcast,
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
        bar = banner("Cost model features + prediction (after pre-scheduling)")
        emit(f"{bar}\n{explain(feats)}\n")
    except Exception as exc:  # noqa: BLE001 - instrumentation must not raise
        emit(f"[SPYRE_DUMP_COST] failed: {exc!r}")
