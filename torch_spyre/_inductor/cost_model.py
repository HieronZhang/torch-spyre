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
  are loaded ONCE and reused across the broadcast dim, so they are counted at their own
  (one-row/-col) DEVICE size -- NOT scaled up to the output size (the rung-6 runs proved
  a core does not re-read the operand per output element), but NOT dropped to zero
  either (it is still a real one-time load). That one load is tiny vs the output, so the
  bcast/mulbcast runs still land ~on the 2-pass latency. They are flagged (``broadcast``
  in :class:`ArgTraffic`) for visibility. (Refinement: with row/col work-splitting the
  operand may be reloaded once per core; counting it a single time is the floor --
  unverified, see open items.)

Byte counts use each arg's DEVICE layout (stick-padded ``device_size``), not the
torch logical shape -- so a reduction's reduced input is naturally full-sized and
stick rounding is captured. REDUCTIONS have an INITIAL (unverified) model: read the
full input at the read-only rate, write the small output, plus a cross-core ring-
combine term when the reduced axis is split across cores. Parameters (``bw_read_gbps``,
``psum_per_elem_ns``, the combine form) are to be calibrated by rung 11. Matmul out
of scope for now.

CALIBRATION NOTE: the golden per-op measurement is the torch.profiler "Self SPYRE"
(sdsc_fused) KERNEL device time. Our SPYRE_PROFILE_SYNC min measured kernel + a non-
deterministic ~20us overhead bucket (the profiler's separate "Memset (Device)" =
host/device setup), so the old ``fill_ns`` ~20us is that OVERHEAD, not kernel cost.
The traffic term alone matches the kernel (gelu[512x1024]: 17.3us kernel ~= 18.9us
traffic). Re-fit ``fill_ns`` against profiler kernel times across sizes (it should
drop toward a small device pipeline-fill).

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
    elems: int  # device element count = prod(dims) (its own one-load size)
    broadcast: bool = False  # loaded once & reused across the broadcast dim
    # DEVICE (stick) shape, e.g. [4, 512, 64]
    dims: list = dataclasses.field(default_factory=list)


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
        # Every HBM-resident arg is counted ONCE at its OWN device size. A broadcast
        # operand already carries its real (one-row/-col) device size in ``elems``, so
        # it is counted a single time -- loaded once and reused across the broadcast
        # dim -- NOT scaled up to the output size, and NOT dropped to zero. Its size is
        # tiny vs the output, so a bcast still lands ~on the 2-pass latency (matching
        # the rung-6 runs), but the one real load is no longer ignored.
        return sum(a.elems for a in self.args if a.mem == "hbm") * self.dtype_bytes

    def lx_bytes(self) -> int:
        return sum(a.elems for a in self.args if a.mem == "lx") * self.dtype_bytes


@dataclasses.dataclass
class CostParams:
    """Fittable parameters. BW in GB/s (numerically == bytes/ns).

    The model predicts the GOLDEN per-kernel device time (torch.profiler "Self SPYRE"),
    NOT our old SPYRE_PROFILE_SYNC min (which folded in a non-deterministic, size-
    scaling Memset/host overhead -- tracked separately, not modeled).

    Fitted from the profiler sweep (examples/run_profile_sweep.sh, section A, fp16):
    - fill ~0 us   -- the kernel has NO fixed term (neg/gelu intercepts -2 to -3 us,
      i.e. ~0; the old ~20 us "fixed" was the overhead bucket, now excluded).
    - BW_HBM ~102 GB/s -- balanced 1R+1W kernel slope (neg 104, gelu 100; R^2 ~ 1.0).
      Kernel time is essentially bytes/BW, linear in I/O size.
    LX traffic ~FREE (rung-4 LX cost below noise; ~29x HBM). Verified: arithmetic-free
    (gelu==neg on kernel time); broadcast/scalar inputs are loaded ONCE at their own
    small device size (rung 6: not re-read per output, but not free either); HBM BW
    shared, core-independent >=2 cores (rung 5).
    PENDING re-anchor on kernel time (sweep sections B-F not yet run):
    - read/write split & the R+W penalty: bw_read/bw_write below are MIN-based (read
      ~176, write ~146, balanced-min ~97 vs the 204.8 LPDDR5 peak) -- re-derive (D).
    - reductions (bw_read, psum) -- section F. Stream-count: rung 7 FALSIFIED a per-
      stream law (4/5-input fused adds match the 1-input rate); plain mul/add ~15-25%
      slow (unexplained). The R+W penalty mechanism needs aiu-smi (see bw note).
    """

    fill_ns: float = 0.0  # golden kernel has ~no fixed term (section A: intercept ~0)
    bw_hbm_gbps: float = 102.0  # balanced 1R+1W KERNEL BW (profiler section A)
    # Reductions (INITIAL, unverified — see rung 11). Read-dominated, so the full
    # input read uses the read-only rate, not the balanced blend. The cross-core
    # ring combine (when the reduced axis is split across k cores) is (k-1) hops,
    # each touching every output element -- mirrors the matmul PSUM term
    # (_PSUM_PER_ELEM_US=1.4e-4 us/elem in work_division.py). Both to be calibrated.
    bw_read_gbps: float = 176.0  # MIN-based (rung-8); re-anchor on kernel time (D/F)
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
            bc = " broadcast (loaded once)" if a.broadcast else ""
            counted = a.elems * o.dtype_bytes if a.mem == "hbm" else 0
            shape = a.dims if a.dims else [a.elems]
            lines.append(
                f"      {a.role:<6} {shape} in {a.mem} = {a.elems} elems x "
                f"{o.dtype_bytes}B = {a.elems * o.dtype_bytes} B"
                f" (hbm counted: {counted} B){bc}"
            )
    t = predict_ops(ops, p)
    lines.append(
        f"  => T = {t / 1000:.2f} us  (fill {p.fill_ns / 1000:.0f}us; pointwise "
        f"@{p.bw_hbm_gbps:.0f}, reductions @{p.bw_read_gbps:.0f} + ring combine)"
    )
    return "\n".join(lines)
