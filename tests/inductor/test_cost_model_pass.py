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

"""Unit tests for the cost-model reporting pass (``cost_model_pass.py``).

The properties guarded here are the ones the report's arithmetic depends on.

* **A kernel is priced once.** ``predict_ops`` is defined over a fused bundle and is
  not additive over its ops -- it de-duplicates external inputs shared inside the
  bundle. Pricing ops separately and summing gives a different number entirely.
  ``_feats`` therefore gives every op the SAME external input ``arg0``, which is what
  makes ``_fused_hbm_bytes`` de-duplicate: with distinct input names the de-dup branch
  never runs and a sum-of-parts implementation would pass every test here.
* **Grouping follows contiguity, not tiling.** ``spyre_fuse_nodes`` fuses every
  contiguous run of Spyre nodes into one bundle, so the report must too. An earlier
  version gave each untiled op its own group and under-predicted a 5-op softmax by 45 %.
* **The per-op column sums back to its kernel total**, because the report prints both.

No Spyre device or backend compiler is required -- ops are injected directly.
"""

import dataclasses

import pytest

import torch_spyre._inductor.cost_model_pass as cmp
from torch_spyre._inductor import config
from torch_spyre._inductor.constants import DEVICE_NAME
from torch_spyre._inductor.cost_model import ArgTraffic, OpFeatures


class _Device:
    def __init__(self, type_):
        self.type = type_


class _FakeLoopInfo:
    def __init__(self, group_id, loop_count):
        self.loop_group_id = group_id
        self.loop_count = loop_count


class _NotComputedBuffer:
    """Stands in for a fallback op the extractor cannot model."""


def _feats(
    name, *, extra_reads=0, write_mb=2, lx_mb=0, shared_input=True, device=DEVICE_NAME
):
    """OpFeatures with a known byte count.

    ``shared_input`` gives the op the external input ``arg0``. That name matters:
    ``_fused_hbm_bytes`` de-duplicates only args whose name starts with ``arg``, so it
    is what makes bundle pricing differ from per-op pricing.
    """
    elems = 512 * 1024  # 1 MB at 2 bytes
    args = []
    if shared_input:
        args.append(ArgTraffic("arg0", "input", "hbm", elems, False, [], []))
    for i in range(extra_reads):
        args.append(ArgTraffic(f"{name}_in{i}", "input", "hbm", elems, False, [], []))
    for i in range(write_mb):
        args.append(ArgTraffic(f"{name}_out{i}", "output", "hbm", elems, False, [], []))
    for i in range(lx_mb):
        args.append(ArgTraffic(f"{name}_lx{i}", "output", "lx", elems, False, [], []))
    f = OpFeatures(
        name=name,
        is_reduction=False,
        out_elems=elems,
        cores=32,
        dtype_bytes=2,
        args=args,
    )
    # The pass breaks a kernel at any op not on the Spyre device, exactly as
    # spyre_fuse_nodes does, so the fixture has to answer get_device().
    f.get_device = lambda d=device: _Device(d) if d else None
    return f


def _ops(feats_list, monkeypatch, loop_infos=None):
    """Route prepared OpFeatures through the real grouping and pricing code."""
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    for f, li in zip(feats_list, loop_infos or [None] * len(feats_list)):
        if li is not None:
            f.loop_info = li
    return feats_list


# --------------------------------------------------------------------------- grouping


def test_empty_graph_is_zero():
    report = cmp.build_report([])
    assert report.total_us == 0.0
    assert report.groups == []


def test_contiguous_ops_form_one_kernel(monkeypatch):
    """This is what spyre_fuse_nodes does -- untiled ops still fuse together."""
    ops = _ops([_feats("a"), _feats("b"), _feats("c")], monkeypatch)
    report = cmp.build_report(ops)
    assert len(report.groups) == 1
    assert report.groups[0].op_names == ["a", "b", "c"]


def test_an_unmodellable_op_breaks_the_kernel(monkeypatch):
    """A non-Spyre node breaks a real bundle; an unmodellable op breaks a group."""
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    report = cmp.build_report(
        [_feats("a"), _feats("b"), _NotComputedBuffer(), _feats("c")]
    )
    assert [g.op_names for g in report.groups] == [["a", "b"], ["c"]]


def test_a_non_spyre_op_breaks_the_kernel(monkeypatch):
    """A CPU op's buffer IS a ComputedBuffer, so the type test alone is not enough.

    spyre_fuse_nodes breaks a bundle at any non-Spyre node; without the device check
    a CPU op would both join the kernel and be priced as Spyre traffic.
    """
    ops = _ops([_feats("a"), _feats("cpu_b", device="cpu"), _feats("c")], monkeypatch)
    report = cmp.build_report(ops)
    assert [g.op_names for g in report.groups] == [["a"], ["c"]]


def test_no_fusion_means_one_kernel_per_op(monkeypatch):
    """With bundle_symbolic_args off the backend does not fuse, so neither do we."""
    ops = _ops([_feats("a"), _feats("b"), _feats("c")], monkeypatch)
    with config.patch({"bundle_symbolic_args": False}):
        report = cmp.build_report(ops)
    assert [g.op_names for g in report.groups] == [["a"], ["b"], ["c"]]


def test_extraction_failure_also_breaks_the_kernel(monkeypatch):
    """An op the extractor rejects must not silently join its neighbours."""

    def _flaky(op):
        if op.name == "bad":
            raise ValueError("cannot model this one")
        return op

    monkeypatch.setattr(cmp, "extract_op_features", _flaky)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    report = cmp.build_report([_feats("a"), _feats("bad"), _feats("c")])
    assert [g.op_names for g in report.groups] == [["a"], ["c"]]


def test_loop_ids_are_recorded_for_labelling(monkeypatch):
    """Tiling no longer decides grouping, but it is still reported."""
    li = _FakeLoopInfo((0,), [8])
    ops = _ops([_feats("a"), _feats("b")], monkeypatch, [li, li])
    group = cmp.build_report(ops).groups[0]
    assert group.loop_group_ids == [(0,)]
    assert group.has_loop
    assert group.loop_trip == 1  # loop_trip comes from OpFeatures, not loop_info


def test_a_malformed_loop_id_does_not_raise(monkeypatch):
    """A bare int where a tuple is expected must not sink the report."""

    class _BadLoopInfo:
        loop_group_id = 1  # not a tuple

    ops = _ops([_feats("a")], monkeypatch, [_BadLoopInfo()])
    report = cmp.build_report(ops)
    assert len(report.groups) == 1
    assert report.groups[0].loop_group_ids == []


# ---------------------------------------------------------------------------- pricing


def test_kernel_is_priced_once_not_per_op(monkeypatch):
    """The kernel total must equal predict_ops over the whole kernel.

    Regression guard against a sum-of-parts implementation. The fixture shares ``arg0``
    across ops so the two genuinely differ -- with distinct input names they coincide
    and this test would pass either way.
    """
    from torch_spyre._inductor.cost_model import CostParams, predict_ops

    feats = [_feats("a"), _feats("b"), _feats("c")]
    ops = _ops(feats, monkeypatch)
    report = cmp.build_report(ops)

    bundle_us = predict_ops(feats, CostParams()) / 1000.0
    parts_us = sum(predict_ops([f], CostParams()) / 1000.0 for f in feats)

    assert report.groups[0].predicted_us == pytest.approx(bundle_us, rel=1e-9)
    # The fixture must actually distinguish the two, or this guards nothing.
    assert parts_us > bundle_us * 1.1


def test_attribution_sums_to_the_kernel_total(monkeypatch):
    """The printed parts must add up to the printed total."""
    ops = _ops(
        [_feats("a", extra_reads=8), _feats("b", extra_reads=2), _feats("c")],
        monkeypatch,
    )
    group = cmp.build_report(ops).groups[0]
    assert sum(o.predicted_us for o in group.ops) == pytest.approx(
        group.predicted_us, rel=1e-9
    )


def test_attribution_follows_main_memory_bytes(monkeypatch):
    """An op with twice the traffic takes twice the share."""
    ops = _ops(
        [
            _feats("big", extra_reads=3, write_mb=0, shared_input=False),
            _feats("small", extra_reads=1, write_mb=0, shared_input=False),
        ],
        monkeypatch,
    )
    by_name = {o.name: o for o in cmp.build_report(ops).groups[0].ops}
    assert by_name["big"].predicted_us == pytest.approx(
        3 * by_name["small"].predicted_us, rel=1e-9
    )


def test_op_with_no_main_memory_traffic_takes_no_share(monkeypatch):
    """An op fused into on-chip memory adds no traffic, so it attracts no time."""
    ops = _ops(
        [
            _feats("streams"),
            _feats("onchip", write_mb=0, lx_mb=4, shared_input=False),
        ],
        monkeypatch,
    )
    by_name = {o.name: o for o in cmp.build_report(ops).groups[0].ops}
    assert by_name["onchip"].predicted_us == 0.0
    assert by_name["onchip"].lx_bytes > 0


def test_total_is_the_sum_over_kernels(monkeypatch):
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    report = cmp.build_report(
        [_feats("a"), _NotComputedBuffer(), _feats("b"), _NotComputedBuffer()]
    )
    assert len(report.groups) == 2
    assert report.total_us == pytest.approx(
        sum(g.predicted_us for g in report.groups), rel=1e-9
    )


def test_one_unpriceable_kernel_does_not_lose_the_others(monkeypatch):
    """A group predict_ops cannot price is dropped; the rest still report."""
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    real = cmp.predict_ops

    def _flaky(feats, params=None):
        if any(f.name == "boom" for f in feats):
            raise ZeroDivisionError("cannot price")
        return real(feats, params)

    monkeypatch.setattr(cmp, "predict_ops", _flaky)
    report = cmp.build_report(
        [
            _feats("a"),
            _NotComputedBuffer(),
            _feats("boom"),
            _NotComputedBuffer(),
            _feats("c"),
        ]
    )
    assert [g.op_names for g in report.groups] == [["a"], ["c"]]
    assert report.total_us > 0.0


# ----------------------------------------------------------------------- post-fusion


class _FakeSchedulerNode:
    def __init__(self, op):
        self.node = op

    def get_nodes(self):
        return [self]


class _FakeLoopNode:
    """A CountedLoopSchedulerNode: node is None, real ops are one level down."""

    def __init__(self, ops):
        self.node = None
        self._inner = [_FakeSchedulerNode(o) for o in ops]

    def get_nodes(self):
        return self._inner


class _FakeFusedNode:
    """A FusedSchedulerNode: get_nodes() does NOT recurse into its members."""

    def __init__(self, members):
        self.node = None
        self._members = members

    def get_nodes(self):
        return self._members


def test_post_fusion_recurses_into_loop_nodes(monkeypatch):
    """A loop node nested in a fused node must not be silently dropped.

    FusedSchedulerNode.get_nodes() returns its members without recursing, and a
    CountedLoopSchedulerNode's own .node is None -- so a flat walk loses every
    coarse-tiled op in the bundle.
    """
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    node = _FakeFusedNode([_FakeLoopNode([_feats("tiled_a"), _feats("tiled_b")])])
    report = cmp.build_report_from_nodes([node])
    assert report.groups[0].op_names == ["tiled_a", "tiled_b"]
    assert report.from_fused_nodes


def test_post_fusion_prices_one_group_per_bundle(monkeypatch):
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    nodes = [
        _FakeFusedNode(
            [_FakeSchedulerNode(_feats("a")), _FakeSchedulerNode(_feats("b"))]
        ),
        _FakeSchedulerNode(_feats("c")),
    ]
    report = cmp.build_report_from_nodes(nodes)
    assert [g.op_names for g in report.groups] == [["a", "b"], ["c"]]


def test_verify_pass_returns_nodes_unchanged(monkeypatch):
    """The post-fusion hook is a node pass: it must return the list it was given."""
    nodes = [object(), object()]
    for mode in ("", "1", "2"):
        with config.patch({"cost_model": mode}):
            assert cmp.verify_against_fused_nodes(nodes) is nodes


def test_verify_pass_returns_nodes_even_when_scoring_fails(monkeypatch):
    monkeypatch.setattr(
        cmp,
        "build_report_from_nodes",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    nodes = [object()]
    with config.patch({"cost_model": "2"}):
        assert cmp.verify_against_fused_nodes(nodes) is nodes


# ---------------------------------------------------------------------------- gating


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", ""),
        ("0", ""),
        ("false", ""),
        ("off", ""),
        ("no", ""),
        ("1", "1"),
        ("true", "1"),
        ("yes", "1"),
        ("on", "1"),
        ("2", "2"),
    ],
)
def test_mode_parsing(value, expected):
    """ "0" must mean OFF. Plain truthiness would enable the pass for SPYRE_DUMP_COST=0."""
    with config.patch({"cost_model": value}):
        assert cmp._mode() == expected


def test_disabled_returns_none_and_does_not_extract(monkeypatch):
    """Off must mean off: no extraction, so leaving it off is free."""
    calls = []
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: calls.append(op) or op)

    class _Graph:
        operations = [_feats("a")]

    for off in ("", "0", "false"):
        with config.patch({"cost_model": off}):
            assert cmp.cost_model_pass(_Graph()) is None
    assert calls == []


def test_enabled_returns_a_report(monkeypatch):
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)

    class _Graph:
        operations = [_feats("a")]

    graph = _Graph()
    with config.patch({"cost_model": "1"}):
        report = cmp.cost_model_pass(graph)
    assert report is not None
    assert report.total_us > 0.0
    assert cmp.LAST_REPORT is report
    assert cmp.LAST_GRAPH_ID == id(graph)


def test_pass_never_raises(monkeypatch):
    """Instrumentation must not be able to break a compilation."""
    monkeypatch.setattr(
        cmp, "build_report", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    class _Graph:
        operations = [1, 2, 3]

    with config.patch({"cost_model": "1"}):
        assert cmp.cost_model_pass(_Graph()) is None


# --------------------------------------------------------------------------- reporting


def test_report_formats_without_error(monkeypatch):
    ops = _ops(
        [_feats("a"), _feats("b", write_mb=0, lx_mb=2, shared_input=False)],
        monkeypatch,
        [_FakeLoopInfo((0,), [8]), _FakeLoopInfo((0,), [8])],
    )
    text = cmp.build_report(ops).format()
    assert "predicted total" in text
    assert "on-chip only" in text
    assert "attribution" in text


def test_empty_report_formats_without_dividing_by_zero():
    assert "predicted total" in cmp.CostReport(total_us=0.0, groups=[]).format()


def test_dataclasses_are_plain_values():
    """The report is a value another pass can hold on to and compare."""
    assert dataclasses.is_dataclass(cmp.CostReport)
    assert dataclasses.is_dataclass(cmp.GroupCost)
    assert dataclasses.is_dataclass(cmp.OpCost)
