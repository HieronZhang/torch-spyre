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


def _input_mem(name: str) -> str:
    """Resolve a read buffer's memory residency (LX vs HBM); default HBM."""
    try:
        from torch._inductor.virtualized import V

        buf = V.graph.get_buffer(name)
        if buf is not None:
            return _mem_of_layout(buf.get_layout())
    except Exception:  # noqa: BLE001 - graph inputs / unresolved -> HBM
        pass
    return "hbm"


def extract_op_features(op) -> OpFeatures:
    """Build OpFeatures for one ComputedBuffer op (best-effort)."""
    data = getattr(op, "data", None)
    is_reduction = getattr(data, "reduction_type", None) is not None
    dtype_bytes = _int(getattr(op.get_dtype(), "itemsize", 2), 2)
    out_size = list(op.get_size())
    out_elems = _prod_ints(out_size)
    cores = _cores(op)

    # Reduction sizing: the full input read = output space x REDUCTION space
    # (Reduction.get_reduction_size() = reduction_ranges). And the reduced axis may
    # be split across cores -> a cross-core ring combine.
    reduction_size = 1
    reduction_cores = 1
    if is_reduction:
        get_rsize = getattr(data, "get_reduction_size", None)
        if callable(get_rsize):
            try:
                reduction_size = _prod_ints(get_rsize())
            except Exception:  # noqa: BLE001 - symbolic/unresolved -> no scaling
                reduction_size = 1
        # Work division splits OUTPUT dims first, then the reduced axis with leftover
        # cores -> the reduced axis is split only when out_elems < cores. Approx k as
        # the cores not absorbed by the output (refine if rung 11 needs the combine).
        if out_elems < cores:
            reduction_cores = max(1, cores // max(1, out_elems))

    args: list = []
    # Output arg.
    args.append(
        ArgTraffic(
            name=op.get_operation_name(),
            role="output",
            mem=_mem_of_layout(op.get_layout()),
            elems=out_elems,
        )
    )
    # Input args, from the op's reads.
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
        # Non-broadcast reads = full output traffic; a reduction's reduced input is
        # read over the full input space (out_elems x reduction_size). Broadcast/
        # scalar reads are cached -> excluded downstream by hbm_bytes().
        read_elems = out_elems
        if is_reduction and not broadcast:
            read_elems = out_elems * reduction_size
        args.append(
            ArgTraffic(
                name=name,
                role="input",
                mem=_input_mem(name),
                elems=read_elems,
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
