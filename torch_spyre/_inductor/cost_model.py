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

    T = fill + hbm_bytes / BW_HBM + lx_bytes / BW_LX

- ``fill`` is the one-time pipeline-fill latency (the fixed term we measured,
  ~15 us). One per kernel/bundle, not per op.
- ``BW_HBM`` / ``BW_LX`` are effective *aggregate* bandwidths (GB/s == bytes/ns).
  Using a single aggregate BW is the "shared HBM" assumption; the SENCORES
  sweep verifies whether core count changes it (add a per-core factor only if a
  benchmark shows the simple model misranks).
- memory traffic counts each tensor-arg's bytes once, attributed to HBM or LX by
  its allocation (LX-placed intermediates don't touch HBM). Broadcast inputs are
  flagged; whether they re-fetch or cache is a measured unknown (``broadcast`` in
  :class:`ArgTraffic`) and defaults to the conservative full count.

FIRST ROUND: pointwise ops only. Reductions are flagged (``is_reduction``) but
their cross-core combine cost is not modeled yet.

Parameters live in :class:`CostParams` and are fit from device measurements
(``examples/bench_*``). Defaults are rough placeholders from early softmax runs.
"""

import dataclasses


@dataclasses.dataclass
class ArgTraffic:
    """Traffic for one tensor argument of an op."""

    name: str
    role: str  # "input" | "output"
    mem: str  # "lx" | "hbm"
    elems: int  # traffic element count (broadcast-adjusted)
    broadcast: bool = False


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
        return sum(a.elems for a in self.args if a.mem == "hbm") * self.dtype_bytes

    def lx_bytes(self) -> int:
        return sum(a.elems for a in self.args if a.mem == "lx") * self.dtype_bytes


@dataclasses.dataclass
class CostParams:
    """Fittable parameters. BW in GB/s (numerically == bytes/ns).

    Fitted from examples/run_cost_model_plan.sh on the run machine (fp16):
    - fill  ~20 us       (rung 1 intercept)
    - BW_HBM ~111 GB/s   (rung 1 slope; 2-stream r/w, >=2 cores, shared & saturated)
    - BW_LX  ~3200 GB/s  (rung 4 chain-depth slope; ~29x HBM, noisy)
    Verified: arithmetic-free for pointwise (rung 2); HBM BW shared, core-independent
    above 2 cores (rung 5). Known gap: 2-input ops run ~15% over the linear traffic
    count (effective BW degrades with stream count) -- refine later.
    """

    fill_ns: float = 20_000.0
    bw_hbm_gbps: float = 111.0
    bw_lx_gbps: float = 3200.0


def predict_ops(ops: list, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a bundle of ops (single pipeline fill)."""
    p = params or CostParams()
    hbm_bytes = sum(o.hbm_bytes() for o in ops)
    lx_bytes = sum(o.lx_bytes() for o in ops)
    return p.fill_ns + hbm_bytes / p.bw_hbm_gbps + lx_bytes / p.bw_lx_gbps


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
            f"  {o.name:<12} out={o.out_elems} cores={o.cores} "
            f"hbm={hbm}B lx={lx}B{red}"
        )
        for a in o.args:
            bc = " broadcast" if a.broadcast else ""
            lines.append(f"      {a.role:<6} {a.mem:<3} {a.elems} elems{bc}")
    t = predict_ops(ops, p)
    lines.append(
        f"  => T = {p.fill_ns:.0f}ns(fill) + {hbm_total}/{p.bw_hbm_gbps}"
        f" + {lx_total}/{p.bw_lx_gbps} = {t / 1000:.2f} us"
    )
    return "\n".join(lines)
