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

FIRST ROUND: pointwise ops only. Reductions are flagged (``is_reduction``) but
their cross-core combine cost is not modeled yet.

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
    Open (single BW=111 is a blend): rung 7 FALSIFIED a stream-count BW law -- 4/5-
    input fused adds (intermediates staged in LX, so still 1 HBM write) match the
    1-input rate, so it is NOT "more streams = slower"; only plain 2-input mul/add is
    anomalously ~15-25% slow. Rung 8 indicates reads are far cheaper than writes
    (read-only ~175 GB/s, 85% of the 204.8 LPDDR5 peak, vs read+write ~98), so a
    read/write-aware BW is the likely refinement -- pending a write-only probe.
    """

    fill_ns: float = 20_000.0
    bw_hbm_gbps: float = 111.0


def predict_ops(ops: list, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a bundle of ops (single fixed term).

    LX-resident traffic is treated as ~free (see :class:`CostParams`), so only HBM
    bytes contribute; LX-placed tensors already drop out of ``hbm_bytes``.
    """
    p = params or CostParams()
    hbm_bytes = sum(o.hbm_bytes() for o in ops)
    return p.fill_ns + hbm_bytes / p.bw_hbm_gbps


def predict_op(op: OpFeatures, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a single op (as its own kernel)."""
    return predict_ops([op], params)


def explain(ops: list, params: CostParams | None = None) -> str:
    """Human-readable breakdown of the prediction for a bundle of ops."""
    p = params or CostParams()
    lines = []
    hbm_total = lx_total = 0
    for o in ops:
        hbm, lx = o.hbm_bytes(), o.lx_bytes()
        hbm_total += hbm
        lx_total += lx
        red = " [reduction: NOT modeled]" if o.is_reduction else ""
        lines.append(
            f"  {o.name:<12} out={o.out_elems} cores={o.cores} hbm={hbm}B lx={lx}B{red}"
        )
        for a in o.args:
            bc = " broadcast (cached: ~free)" if a.broadcast else ""
            lines.append(f"      {a.role:<6} {a.mem:<3} {a.elems} elems{bc}")
    t = predict_ops(ops, p)
    lines.append(
        f"  => T = {p.fill_ns:.0f}ns(fill) + {hbm_total}/{p.bw_hbm_gbps}"
        f"  (lx {lx_total}B: ~free) = {t / 1000:.2f} us"
    )
    return "\n".join(lines)
