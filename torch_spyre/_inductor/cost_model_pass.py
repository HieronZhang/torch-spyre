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

"""Predicted-runtime reporting over the pre-scheduling loop-level IR.

Runs immediately after ``CustomPreSchedulingPasses`` has finished, where the
loop-level IR is final: layouts resolved, restickify inserted, coarse-tiling applied,
work division committed, scratchpad placed. It partitions the graph into the kernels
the backend will build, prices each with the analytical model in ``cost_model.py``,
and returns a :class:`CostReport` carrying the program total plus a breakdown.

The report is a *value*, not just a printout, so another pass or an external tool can
compare two candidate plans without compiling and running either. That is the point:
``report_a.total_us`` vs ``report_b.total_us`` is a cheap plan comparison.

NOT related to ``work_division.cost_model_matmul_division``, which is a separate model
used to choose a matmul work division. This module is the analytical whole-program
runtime model documented in ``docs/source/compiler/cost_model.md``.

Disabled by default. When the flag is off this module does no work at all -- it returns
before touching the graph -- so leaving it off costs one attribute read per compilation.

HOW OPS ARE GROUPED, AND WHY IT MATTERS
---------------------------------------
``predict_ops`` is defined over ONE FUSED KERNEL and is deliberately not additive over
its ops: it de-duplicates external inputs shared inside a bundle and its turnaround and
overlap terms are ``min``/``max`` reductions over bundle totals. Pricing ops separately
and adding is therefore simply a different quantity -- measured across 521 recorded
multi-op bundles the difference ranges from -94 % to +33 %. It is zero on 32 % of them
(no shared external input, so the de-dup never fires) and, where it differs at all, an
UNDER-count about seven times in ten. So the grouping has to match what the backend
actually fuses.

``spyre_fuse_nodes`` (``fusion.py``) accumulates every *contiguous run* of Spyre nodes
into one bundle, with no size limit; only a non-Spyre node breaks the run. This module
mirrors that: contiguous modellable ops form one group. An earlier version made every
untiled op its own group, which under-predicted a 5-op softmax by 45 % because the real
kernel is one bundle, not five.

Coarse-tiling ``loop_group_id`` is still read, but for *labelling* the loop structure
inside a group -- not for deciding group boundaries.
"""

from __future__ import annotations

import dataclasses

from torch._inductor.ir import ComputedBuffer

from . import config
from .constants import DEVICE_NAME
from .cost_model import CostParams, predict_ops
from .dump_cost_model import extract_op_features
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("cost_model_pass")

#: Truthy spellings, matching ``dump_cost_model.cost_dump_enabled`` so that "0" means
#: off. Gating on plain truthiness would enable the pass for SPYRE_DUMP_COST=0.
_ON = {"1", "2", "true", "yes", "on"}

#: Report from the most recent pre-scheduling run, tagged with the id of the graph it
#: came from. The post-fusion check needs it, and the two pipelines are separate objects
#: with no reference to each other; this mirrors ``dump_cost_model.LAST_IO``, the
#: existing convention. The id guards against a nested lowering silently pairing one
#: graph's estimate with another graph's bundles.
LAST_REPORT: "CostReport | None" = None
LAST_GRAPH_ID: int | None = None


def _mode() -> str:
    """ "" (off), "1" (report), or "2" (report + post-fusion check)."""
    raw = str(config.cost_model or "").strip().lower()
    if raw not in _ON:
        return ""
    return "2" if raw == "2" else "1"


@dataclasses.dataclass
class OpCost:
    """One operation's share of its group's predicted time.

    ``predicted_us`` is an ATTRIBUTION of the group total, not a standalone prediction
    for this op -- pricing ops separately would give a different number entirely.
    """

    name: str
    loop_group_id: tuple[int, ...] | None
    hbm_bytes: int
    lx_bytes: int
    predicted_us: float


@dataclasses.dataclass
class GroupCost:
    """One kernel: a contiguous run of ops the backend will fuse into one bundle."""

    index: int
    op_names: list[str]
    #: Distinct coarse-tiling loop groups represented in this kernel, for labelling.
    loop_group_ids: list[tuple[int, ...]]
    loop_trip: int
    predicted_us: float
    ops: list[OpCost]

    @property
    def has_loop(self) -> bool:
        return bool(self.loop_group_ids)


@dataclasses.dataclass
class CostReport:
    """Predicted runtime for one compiled graph.

    ``total_us`` is the headline: the sum over kernels. It is the number another
    component should compare between two candidate plans.
    """

    total_us: float
    groups: list[GroupCost]
    #: True when built from real post-fusion bundles rather than estimated ones.
    from_fused_nodes: bool = False

    @property
    def n_modelled_ops(self) -> int:
        return sum(len(g.op_names) for g in self.groups)

    def format(self) -> str:
        kind = "bundle" if self.from_fused_nodes else "kernel"
        lines = [
            f"predicted total: {self.total_us:10.1f} us over "
            f"{len(self.groups)} {kind}(s), {self.n_modelled_ops} op(s)",
            "",
            f"  {kind:>10}  {'loops':>10}  {'trip':>5}  {'predicted us':>13}  ops",
        ]
        for g in sorted(self.groups, key=lambda x: -x.predicted_us):
            loops = (
                ",".join("/".join(str(i) for i in gid) for gid in g.loop_group_ids)
                if g.has_loop
                else "-"
            )
            lines.append(
                f"  {g.index:>10}  {loops:>10}  {g.loop_trip:>5}  "
                f"{g.predicted_us:>13.1f}  {len(g.op_names)}"
            )
            for o in sorted(g.ops, key=lambda x: -x.predicted_us):
                # An op whose output stays on chip moves no main-memory bytes, so it
                # attracts no share. Showing the on-chip bytes makes that 0.0 read as
                # "this op was fused away", not "this op is free".
                where = (
                    f"{o.hbm_bytes / 1e6:.1f} MB"
                    if o.hbm_bytes
                    else f"on-chip only, {o.lx_bytes / 1e6:.1f} MB"
                )
                lines.append(
                    f"  {'':>10}  {'':>10}  {'':>5}  {o.predicted_us:>13.1f}  "
                    f"  {o.name} ({where})"
                )
        lines += [
            "",
            "  Each kernel is priced ONCE, as a bundle. The per-op column splits that",
            "  total by each op's share of the main-memory bytes: an attribution, not a",
            "  set of independent predictions. It carries no compute term, so it",
            "  misattributes a compute-bound kernel, and because the weights are per-op",
            "  while the total de-duplicates shared inputs, a share can be off by up to a",
            "  third of itself. An op whose output stays on chip shows 0.0 -- it adds no",
            "  traffic of its own.",
        ]
        return "\n".join(lines)


def _is_modellable(op) -> bool:
    """True when ``op`` belongs in a Spyre bundle.

    Both halves matter. ``spyre_fuse_nodes`` breaks a bundle at any node that is not
    on the Spyre device, and a CPU op's buffer IS a ``ComputedBuffer`` -- so testing
    the type alone neither breaks the run nor stops the op being priced as Spyre
    traffic. ``SpyreConstantFallback``/``SpyreEmptyFallback`` are ExternKernels and
    are excluded by the type test.
    """
    if not isinstance(op, ComputedBuffer):
        return False
    try:
        device = op.get_device()
    except Exception:  # noqa: BLE001 - best-effort; treat as a boundary
        return False
    return device is not None and device.type == DEVICE_NAME


def _loop_group_id(op) -> tuple[int, ...] | None:
    gid = getattr(getattr(op, "loop_info", None), "loop_group_id", None)
    try:
        return tuple(gid) if gid else None
    except TypeError:  # a malformed id (e.g. a bare int) must not sink the report
        return None


def _price(feats, index, params) -> GroupCost | None:
    """Price one kernel. Returns None if the model cannot price it."""
    try:
        predicted_us = predict_ops(feats, params) / 1000.0
    except Exception:  # noqa: BLE001 - a bad group must not lose the whole report
        logger.warning("cost model could not price kernel %d; skipping", index)
        return None

    # Attribute by main-memory bytes. The weights are each op's OWN byte count while
    # the total de-duplicates inputs shared across the kernel, so the split is not
    # self-consistent: measured on recorded softmax bundles an op's share can be off by
    # up to a third of itself (e.g. 33.3 % where a consistent split gives 25.0 %), and
    # the error runs both ways -- de-dup shifts weight off the shared-input readers and
    # onto the op that owns the write. Kept because a consistent split would need the
    # reader set per input, which OpFeatures does not carry; flagged in format().
    weights = [max(0, f.hbm_bytes()) for f in feats]
    total = sum(weights)
    ops = [
        OpCost(
            name=f.name,
            loop_group_id=getattr(f, "_loop_group_id", None),
            hbm_bytes=w,
            lx_bytes=max(0, f.lx_bytes()),
            predicted_us=predicted_us
            * (w / total if total > 0 else 1.0 / max(1, len(feats))),
        )
        for f, w in zip(feats, weights)
    ]
    gids: list[tuple[int, ...]] = []
    for f in feats:
        gid = getattr(f, "_loop_group_id", None)
        if gid is not None and gid not in gids:
            gids.append(gid)
    return GroupCost(
        index=index,
        op_names=[f.name for f in feats],
        loop_group_ids=gids,
        loop_trip=max((getattr(f, "loop_trip", 1) or 1) for f in feats),
        predicted_us=predicted_us,
        ops=ops,
    )


def build_report(operations: list, params: CostParams | None = None) -> CostReport:
    """Price ``operations`` kernel by kernel, mirroring how the backend fuses.

    Split out from the pass so it can be unit-tested without a GraphLowering.
    """
    params = params or CostParams()

    # Contiguous runs of modellable ops become one kernel each, which is what
    # spyre_fuse_nodes does. Anything it cannot model breaks the run, exactly as a
    # non-Spyre node breaks a real bundle.
    runs: list[list] = []
    current: list = []
    # When symbolic bundle args are off the backend does not fuse at all
    # (fusion.py returns the node list untouched), so every op is its own kernel.
    fuses = bool(getattr(config, "bundle_symbolic_args", True))

    for op in operations:
        feats = None
        if _is_modellable(op):
            try:
                feats = extract_op_features(op)
                feats._loop_group_id = _loop_group_id(op)
            except Exception:  # noqa: BLE001 - skip ops the extractor cannot model
                feats = None
        if feats is None:
            if current:
                runs.append(current)
                current = []
            continue
        current.append(feats)
        if not fuses:  # no fusion -> one kernel per op
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    groups = [g for i, r in enumerate(runs) if (g := _price(r, i, params)) is not None]
    return CostReport(total_us=sum(g.predicted_us for g in groups), groups=groups)


def _iter_ops(node):
    """Every ``ComputedBuffer`` under a scheduler node, recursing into loop nodes.

    ``FusedSchedulerNode.get_nodes()`` returns its members without recursing, and a
    ``CountedLoopSchedulerNode`` nested inside one has ``node is None`` -- so a
    non-recursive walk silently drops every coarse-tiled op. ``hbm_pool_planning``
    has the same recursion for the same reason.
    """
    for snode in node.get_nodes() if hasattr(node, "get_nodes") else [node]:
        op = getattr(snode, "node", None)
        if _is_modellable(op):
            yield op
        elif snode is not node and hasattr(snode, "get_nodes"):
            yield from _iter_ops(snode)


def build_report_from_nodes(
    nodes: list, params: CostParams | None = None
) -> CostReport:
    """Price already-fused scheduler nodes: one kernel per real bundle.

    After ``spyre_fuse_nodes`` each node IS one SuperDSC bundle, which is exactly what
    ``predict_ops`` is defined over. This is therefore the correct grouping, against
    which ``build_report``'s contiguity estimate can be checked.
    """
    params = params or CostParams()
    groups: list[GroupCost] = []
    for index, node in enumerate(nodes):
        feats = []
        for op in _iter_ops(node):
            try:
                f = extract_op_features(op)
            except Exception:  # noqa: BLE001 - skip ops the extractor cannot model
                continue
            f._loop_group_id = _loop_group_id(op)
            feats.append(f)
        if feats and (g := _price(feats, index, params)) is not None:
            groups.append(g)
    return CostReport(
        total_us=sum(g.predicted_us for g in groups),
        groups=groups,
        from_fused_nodes=True,
    )


def _current_graph_id() -> int | None:
    """id() of the graph being lowered, or None when it cannot be determined.

    The post-fusion pipeline receives scheduler nodes, not the graph, so the identity
    has to come from Inductor's ambient state.
    """
    try:
        from torch._inductor.virtualized import V

        return id(V.graph)
    except Exception:  # noqa: BLE001 - no ambient graph
        return None


def verify_against_fused_nodes(nodes: list) -> list:
    """Re-score post-fusion and report how far the pre-scheduling estimate was off.

    Runs only in mode "2". Returns ``nodes`` untouched -- this is a node-pipeline pass
    and must return the list it was given.

    The point is honesty: ``build_report`` estimates kernel boundaries from op
    contiguity, but the real bundles are only known here. This measures the gap rather
    than asserting it is small.
    """
    if _mode() != "2":
        return nodes

    try:
        after = build_report_from_nodes(nodes)
        from .dump_common import banner, emit

        lines = [
            banner("Cost model: re-scored against real fusion bundles"),
            after.format(),
            "",
        ]
        # Only compare against an estimate from THIS compilation. A nested lowering
        # overwrites LAST_REPORT, and CustomPreSchedulingPasses early-returns on a
        # non-Spyre graph without clearing it, so an unguarded read can pair one
        # graph's estimate with another graph's bundles.
        before = LAST_REPORT if LAST_GRAPH_ID == _current_graph_id() else None
        if before is not None and before.total_us > 0:
            delta = after.total_us - before.total_us
            lines.append(
                f"  pre-scheduling estimate {before.total_us:10.1f} us over "
                f"{len(before.groups)} kernel(s)\n"
                f"  real fusion bundles     {after.total_us:10.1f} us over "
                f"{len(after.groups)} bundle(s)\n"
                f"  difference              {delta:+10.1f} us "
                f"({100.0 * delta / before.total_us:+.1f} %)"
            )
        emit("\n".join(lines) + "\n")
    except Exception as exc:  # noqa: BLE001 - instrumentation must not raise
        logger.warning("post-fusion cost re-score failed: %r", exc)

    return nodes


def cost_model_pass(graph) -> CostReport | None:
    """Predict this graph's runtime; ``None`` when the model is disabled.

    Read-only: the graph is never mutated. Called after the pre-scheduling pipeline,
    where the loop-level IR is final.

    Returns the report so a caller can use it -- ``CustomPreSchedulingPasses`` stores it
    as ``last_cost_report`` for exactly that reason.
    """
    global LAST_REPORT, LAST_GRAPH_ID
    LAST_REPORT, LAST_GRAPH_ID = None, None
    if not _mode():
        return None

    try:
        report = build_report(getattr(graph, "operations", None) or [])
    except Exception as exc:  # noqa: BLE001 - instrumentation must not raise
        # A sibling dump hook once raised ImportError here and broke every compile.
        # Reporting must never be able to do that.
        logger.warning("cost model pass failed: %r", exc)
        return None

    try:
        from .dump_common import banner, emit

        emit(
            banner("Cost model: predicted runtime (after pre-scheduling)")
            + "\n"
            + report.format()
            + "\n"
        )
    except Exception as exc:  # noqa: BLE001 - printing must not raise either
        logger.warning("cost model report could not be emitted: %r", exc)

    LAST_REPORT, LAST_GRAPH_ID = report, _current_graph_id() or id(graph)
    return report
