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

"""A simple, high-level analytical cost model for Spyre kernels.

Goal: predict *relative* device latency from the "after-pre-scheduling"
LoopLevel IR to guide higher-level optimization. Deliberately NOT a simulator.

Model (per fused bundle / single-op kernel):

    T = fill + hbm_bytes / BW_HBM        (LX-resident traffic treated as ~free)

- ``fill`` is the fixed per-kernel cost (~20 us): pipeline fill/drain + device
  setup + ~7 us host dispatch/sync residue. One per kernel/bundle, not per op.
- ``BW_HBM`` is the effective *aggregate* HBM bandwidth (GB/s == bytes/ns).
  Using a single aggregate BW is the "shared HBM" assumption; the SENCORES
  sweep verifies whether core count changes it (add a per-core factor only if a
  benchmark shows the simple model misranks).
- memory traffic counts each tensor-arg's bytes once, attributed to HBM or LX by
  its allocation. LX-placed tensors don't touch HBM, and their LX traffic is treated
  as ~free (the measured per-pass LX cost is below run-to-run noise). Broadcast inputs
  are cached on-chip (loaded once, reused across the broadcast dim) and so add ~no HBM
  traffic -- verified by the rung-6 bcast/mulbcast runs, which land on the 2-pass
  latency. They are flagged (``broadcast`` in :class:`ArgTraffic`) and excluded from
  ``hbm_bytes``.

Pointwise is calibrated. REDUCTIONS have an INITIAL (unverified) model: read the
full input (out_elems x reduction_size) at the read-only rate, write the small
output, plus a cross-core ring-combine term when the reduced axis is split across
cores. Parameters (``bw_read_gbps``, ``psum_per_elem_ns``, the combine form) are to
be calibrated by rung 11. Matmul is out of scope for now.

Parameters live in :class:`CostParams`, calibrated from device measurements
(``examples/run_cost_model_plan.sh``).
"""

import dataclasses


@dataclasses.dataclass
class ArgTraffic:
    """Traffic for one tensor argument of an op."""

    name: str
    role: str  # "input" | "output"
    mem: str  # "lx" | "hbm"
    elems: int  # full element count for this arg (pre-broadcast-discount)
    broadcast: bool = False  # cached on-chip -> excluded from hbm_bytes()


@dataclasses.dataclass
class OpFeatures:
    """Cost-relevant features of one LoopLevel-IR op."""

    name: str  # origin op name (e.g. "gelu", "mul", "sub")
    is_reduction: bool
    out_elems: int
    cores: int
    dtype_bytes: int
    args: list  # list[ArgTraffic]
    reduction_cores: int = 1  # cores splitting the REDUCED axis (1 = none → no combine)

    def hbm_bytes(self) -> int:
        # Broadcast inputs are cached on-chip (loaded once, reused across the
        # broadcast dim), so they add ~no HBM traffic. Verified on device: the
        # rung-6 bcast/mulbcast runs land on the 2-pass (no-broadcast) latency,
        # i.e. the [1,N] operand is effectively free. So exclude broadcast args.
        return (
            sum(a.elems for a in self.args if a.mem == "hbm" and not a.broadcast)
            * self.dtype_bytes
        )

    def lx_bytes(self) -> int:
        return sum(a.elems for a in self.args if a.mem == "lx") * self.dtype_bytes


@dataclasses.dataclass
class CostParams:
    """Fittable parameters. BW in GB/s (numerically == bytes/ns).

    Fitted from examples/run_cost_model_plan.sh on the run machine (fp16):
    - fill  ~20 us       (rung 1 intercept; op-independent)
    - BW_HBM ~111 GB/s   (rung 1 slope; 2-stream r/w, >=2 cores, shared & saturated)
    LX traffic is treated as ~FREE (no BW_LX term): the rung-4 chain showed the
    per-op LX cost (~1 us) sits below the run-to-run measurement noise (~5 us), so a
    precise BW_LX can't be resolved and LX-resident tensors barely affect latency.
    Qualitatively LX is ~29x HBM; quantitatively we drop the term.
    Verified: arithmetic-free for pointwise (rung 2); HBM BW shared, core-independent
    above 2 cores (rung 5); broadcast inputs cached/free (rung 6).
    BW_HBM=111 is the BALANCED read+write rate. Rung 8 (measured): read-only ~176 GB/s
    (86% of the 204.8 LPDDR5 peak; _HBM_BW_GBS in work_division.py), write-only ~146,
    balanced 1R+1W ~97. Mixing reads+writes ~halves throughput, so pointwise caps ~100.
    A read/write-aware BW (reads 176 / writes 146) is a future refinement, worth it only
    if the blend misranks read-heavy ops (reductions).
    Open: rung 7 FALSIFIED a stream-count BW law -- 4/5-input fused adds keep their
    intermediate in LX (still 1 HBM write) and match the 1-input rate, so NOT
    "more streams = slower". Only plain 2-input mul/add is ~15-25% slow (unexplained).
    The penalty's mechanism (turnaround? half-duplex? shared bus) needs an aiu-smi
    capture -- see notes/bandwidth_turnaround_experiment.md.
    """

    fill_ns: float = 20_000.0
    bw_hbm_gbps: float = 111.0
    # Reductions (INITIAL, unverified — see rung 11). Read-dominated, so the full
    # input read uses the read-only rate, not the balanced blend. The cross-core
    # ring combine (when the reduced axis is split across k cores) is (k-1) hops,
    # each touching every output element -- mirrors the matmul PSUM term
    # (_PSUM_PER_ELEM_US=1.4e-4 us/elem in work_division.py). Both to be calibrated.
    bw_read_gbps: float = 176.0  # rung-8 read-only asymptote
    psum_per_elem_ns: float = 0.14  # 1.4e-4 us/elem/hop, from the matmul model


def predict_ops(ops: list, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a bundle of ops (single fixed term).

    LX-resident traffic is treated as ~free (see :class:`CostParams`), so only HBM
    bytes contribute. Pointwise ops use the balanced read+write ``bw_hbm_gbps``;
    REDUCTIONS are read-dominated, so they use the read-only ``bw_read_gbps`` plus a
    cross-core ring-combine term when the reduced axis is split across cores.
    """
    p = params or CostParams()
    t = p.fill_ns
    for o in ops:
        if o.is_reduction:
            t += o.hbm_bytes() / p.bw_read_gbps
            t += max(0, o.reduction_cores - 1) * o.out_elems * p.psum_per_elem_ns
        else:
            t += o.hbm_bytes() / p.bw_hbm_gbps
    return t


def predict_op(op: OpFeatures, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a single op (as its own kernel)."""
    return predict_ops([op], params)


def explain(ops: list, params: CostParams | None = None) -> str:
    """Human-readable breakdown of the prediction for a bundle of ops."""
    p = params or CostParams()
    lines = []
    for o in ops:
        hbm, lx = o.hbm_bytes(), o.lx_bytes()
        if o.is_reduction:
            combine = max(0, o.reduction_cores - 1) * o.out_elems * p.psum_per_elem_ns
            red = (
                f" [reduction: read@{p.bw_read_gbps:.0f}, "
                f"combine {combine:.0f}ns (k={o.reduction_cores})]"
            )
        else:
            red = ""
        lines.append(
            f"  {o.name:<12} out={o.out_elems} cores={o.cores} hbm={hbm}B lx={lx}B{red}"
        )
        for a in o.args:
            bc = " broadcast (cached: ~free)" if a.broadcast else ""
            lines.append(f"      {a.role:<6} {a.mem:<3} {a.elems} elems{bc}")
    t = predict_ops(ops, p)
    lines.append(
        f"  => T = {t / 1000:.2f} us  (fill {p.fill_ns / 1000:.0f}us; pointwise "
        f"@{p.bw_hbm_gbps:.0f}, reductions @{p.bw_read_gbps:.0f} + ring combine)"
    )
    return "\n".join(lines)
