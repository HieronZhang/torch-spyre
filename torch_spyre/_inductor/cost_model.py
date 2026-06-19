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

    T = fill + (R + W) / BW_PEAK + alpha * min(R, W)

  where R = HBM bytes READ (inputs), W = HBM bytes WRITTEN (outputs). LX-resident
  traffic is treated as ~free.

- ``fill`` ~= 0: the golden kernel has no fixed term (section-A intercept ~0; the old
  ~20us "fixed" was a separate non-deterministic Memset/host-setup bucket, not kernel).
- ``BW_PEAK`` ~150 GB/s (== bytes/ns) is the PEAK HBM bandwidth, reached when traffic is
  one-directional (read-only or write-only). HBM is a shared bus that must turn around
  between reading and writing, so a kernel doing BOTH pays a penalty on the overlap.
- ``alpha * min(R, W)`` is that read/write turnaround penalty. min(R,W) is the overlap:
  0 for pure read or pure write (no switching), maximal at a balanced 1R+1W. This gives
  the measured V-shaped effective BW -- ~150 read-only, dipping to ~105 at balanced,
  back up for write-only -- with one extra constant instead of a second bandwidth.
  Verified on the B-F profiler sweep: ~2% error on core pointwise + reductions, ~7%
  overall (turnaround) vs ~11% for an additive two-rate. Using a single aggregate
  BW_PEAK is the "shared HBM" assumption (rung-5: core-independent for >=2 cores).
- memory traffic counts each tensor-arg's bytes once, attributed to HBM or LX by
  its allocation. LX-placed tensors don't touch HBM, and their LX traffic is treated
  as ~free (the measured per-pass LX cost is below run-to-run noise). Broadcast inputs
  are loaded ONCE and reused across the broadcast dim, so they are counted at their own
  (one-row/-col) DEVICE size -- NOT scaled up to the output size (the rung-6 runs proved
  a core does not re-read the operand per output element), but NOT dropped to zero
  either (it is still a real one-time load). That one load is tiny vs the output, so the
  bcast/mulbcast runs still land ~on the 2-pass latency. They are flagged (``broadcast``
  in :class:`ArgTraffic`) for visibility. The "once, not per core" count is VERIFIED on
  device (rung-G reload probe, cores=32, R=64): bcast (b[1,C]) ~= bcastcol (b[R,1]) ~=
  30-33us, both far below the full 3-pass add (52us). A per-core reload would have added
  ~cores*C and pushed bcast up toward add; it did not -- so the operand costs a single
  load regardless of how the work splits across cores.

Byte counts use each arg's DEVICE layout (stick-padded ``device_size``), not the torch
logical shape -- so a reduction's reduced input is naturally full-sized and stick
rounding is captured. REDUCTIONS need no special bandwidth: a reduction is just a kernel
with a tiny WRITE (the small output), so the turnaround penalty vanishes (min(R,W)~0)
and T ~= R / BW_PEAK -- the read-only rate -- automatically. ``is_reduction`` survives
only to add a cross-core ring-combine term (``psum_per_elem_ns``) when the reduced axis
is split across cores. Matmul out of scope for now.

KNOWN systematic biases (B-F sweep; consistent per-category, so within-kind ranking is
safe, cross-kind comparisons can be off ~15-20%):
- broadcast pointwise (bcast) runs ~17% FASTER than the model (off the V-curve; cause
  open) -- the model over-predicts their time.
- write-only runs ~16% off (BW_PEAK treats writes like reads; writes are a bit slower).
- high fan-in fused adds (add3/add4) ~8% (LX intermediates not perfectly free).
- sumcol (reducing the outer/partitioned axis) ~19% (access-pattern, not ring-combine).

CALIBRATION NOTE: the golden per-op measurement is the torch.profiler "Self SPYRE"
(sdsc_fused) KERNEL device time -- NOT our old SPYRE_PROFILE_SYNC min (which folded in a
non-deterministic Memset/host-setup bucket, the source of the obsolete ~20us fill).

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
    # Coarse-tiling loop multiplier on this arg's bytes. 1 for a normal arg or an
    # ADVANCING tiled arg (it walks the full tensor once across the loop, so its full
    # device_size already covers all tiles). L (= loop trip count) for a FIXED arg held
    # at one address across the loop (a per-tile accumulator re-read/written each
    # iteration). LX-resident args are ~free regardless (excluded from read/write).
    loop_factor: int = 1


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
    loop_trip: int = 1  # coarse-tiling loop trip count for this op (prod of loop_count)

    def read_bytes(self) -> int:
        """HBM bytes READ (input args). Each HBM arg is counted at its own device size,
        scaled by ``loop_factor`` (L for a per-tile accumulator re-read every iteration,
        1 for an advancing tiled arg or a normal arg). A broadcast operand carries its
        real (one-row/-col) ``elems`` -- loaded once, NOT scaled to the output.
        """
        return (
            sum(
                a.elems * a.loop_factor
                for a in self.args
                if a.mem == "hbm" and a.role == "input"
            )
            * self.dtype_bytes
        )

    def write_bytes(self) -> int:
        """HBM bytes WRITTEN (output args), scaled by ``loop_factor``."""
        return (
            sum(
                a.elems * a.loop_factor
                for a in self.args
                if a.mem == "hbm" and a.role == "output"
            )
            * self.dtype_bytes
        )

    def hbm_bytes(self) -> int:
        """Total HBM traffic = read + write (kept for the dump / LAST_IO totals)."""
        return self.read_bytes() + self.write_bytes()

    def lx_bytes(self) -> int:
        return sum(a.elems for a in self.args if a.mem == "lx") * self.dtype_bytes


@dataclasses.dataclass
class CostParams:
    """Fittable parameters for ``T = fill + (R+W)/BW_PEAK + alpha*min(R,W)``.

    Predicts the GOLDEN per-kernel device time (torch.profiler "Self SPYRE"). Fitted on
    the B-F profiler sweep (examples/run_profile_sweep.sh, fp16):
    - fill ~0       -- no fixed kernel term (section A intercept ~0).
    - BW_PEAK ~150 GB/s (== bytes/ns) -- the one-directional peak; read-only reductions
      and read probes land at ~145-155.
    - alpha ~0.00574 ns/byte -- the read/write turnaround penalty, calibrated so a
      balanced 1R+1W neg (R=W) lands at its measured ~105 GB/s effective:
      2/(2/BW_PEAK + alpha) = 105. min(R,W) is the read/write overlap (0 for
      one-directional traffic, maximal at balanced) -> reproduces the V-shaped
      effective BW. ~2% error on core ops, ~7% overall (see module docstring biases).
    LX traffic ~FREE (rung-4 below noise). Verified: arithmetic-free (gelu/exp == neg);
    broadcast operand loaded ONCE (rung-G, not per core); HBM BW shared / core-
    independent >=2 cores (rung 5).
    """

    fill_ns: float = 0.0  # golden kernel has ~no fixed term (section A: intercept ~0)
    bw_peak_gbps: float = 150.0  # one-directional peak HBM BW (read-only / write-only)
    # Read/write turnaround penalty (ns per overlapping byte). HBM is a shared bus that
    # must switch between read and write; the cost falls on the overlap min(R,W). Solved
    # from balanced neg (eff 105): alpha = 2/105 - 2/BW_PEAK.
    rw_turnaround_ns_per_byte: float = 0.00574
    # Reduction cross-core ring combine: (k-1) hops each touching every output element,
    # mirrors the matmul PSUM term (_PSUM_PER_ELEM_US in work_division.py). Weak (pinned
    # by sumall; sumcol's ~19% miss looks like an access-pattern effect, not this).
    psum_per_elem_ns: float = 0.14
    # Coarse-tiling per-iteration loop overhead (ns/iteration). With unroll_loops=True
    # the loop unrolls into L body copies in one bundle; each adds a fixed cost beyond
    # its memory traffic. CALIBRATED ~860 ns/tile from the P2a K-sweep (ctsum B=2048
    # D=512, tiles 2..16): unmodeled = (kernel - pred) fits 1.88us + 0.864*K, so the
    # K-slope is the loop overhead (the 1.88us intercept is the dim0-reduction access
    # penalty -- a separate sumcol-like bias, NOT loop cost; the D-sweep gives the same
    # K-slope, so it is data-independent). Tiling a STANDALONE reduction is thus slower
    # (no LX win); the payoff is fused chains keeping intermediates in LX (not scoped).
    c_loop_ns: float = 860.0


def predict_ops(ops: list, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a bundle of ops (one fused kernel).

    ``T = fill + (R+W)/BW_PEAK + alpha*min(R,W) + c_loop*L`` where R/W are the bundle's
    total HBM read/write bytes (LX ~free), already loop-scaled per arg (see
    ArgTraffic.loop_factor), and ``L`` is the tiling trip count (1 when not tiled).
    R and W are summed over the whole bundle before the turnaround term, since a fused
    kernel interleaves all its reads and writes on a shared bus. Reductions add a
    cross-core ring-combine term, charged once PER TILE (x loop_trip).
    """
    p = params or CostParams()
    r = sum(o.read_bytes() for o in ops)
    w = sum(o.write_bytes() for o in ops)
    t = p.fill_ns + (r + w) / p.bw_peak_gbps + p.rw_turnaround_ns_per_byte * min(r, w)
    for o in ops:
        if o.is_reduction:
            combine = max(0, o.reduction_cores - 1) * o.out_elems * p.psum_per_elem_ns
            t += combine * o.loop_trip
    # Coarse-tiling loop overhead: L body dispatches for the bundle's tiled loop (0 when
    # nothing is tiled). L = max op trip count; looped ops agree, the fill op (trip=1)
    # does not raise it.
    loop_trip = max((o.loop_trip for o in ops), default=1)
    if loop_trip > 1:
        t += p.c_loop_ns * loop_trip
    return t


def predict_op(op: OpFeatures, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a single op (as its own kernel)."""
    return predict_ops([op], params)


def explain(ops: list, params: CostParams | None = None) -> str:
    """Human-readable breakdown of the prediction for a bundle of ops."""
    p = params or CostParams()
    lines = []
    for o in ops:
        r, w, lx = o.read_bytes(), o.write_bytes(), o.lx_bytes()
        if o.is_reduction:
            combine = max(0, o.reduction_cores - 1) * o.out_elems * p.psum_per_elem_ns
            red = f" [reduction: combine {combine:.0f}ns (k={o.reduction_cores})]"
        else:
            red = ""
        loop = f" loop_trip={o.loop_trip}" if o.loop_trip > 1 else ""
        lines.append(
            f"  {o.name:<12} read={r}B write={w}B lx={lx}B{loop}{red}"
        )
        for a in o.args:
            bc = " broadcast (loaded once)" if a.broadcast else ""
            lf = f" xL={a.loop_factor}" if a.loop_factor > 1 else ""
            counted = a.elems * a.loop_factor * o.dtype_bytes if a.mem == "hbm" else 0
            shape = a.dims if a.dims else [a.elems]
            lines.append(
                f"      {a.role:<6} {shape} in {a.mem} = {a.elems} elems x "
                f"{o.dtype_bytes}B = {a.elems * o.dtype_bytes} B"
                f" (hbm counted: {counted} B){lf}{bc}"
            )
    R = sum(o.read_bytes() for o in ops)
    W = sum(o.write_bytes() for o in ops)
    base = (R + W) / p.bw_peak_gbps
    turn = p.rw_turnaround_ns_per_byte * min(R, W)
    loop_trip = max((o.loop_trip for o in ops), default=1)
    loop_us = (p.c_loop_ns * loop_trip / 1000) if loop_trip > 1 else 0.0
    t = predict_ops(ops, p)
    extra = f" + loop {loop_us:.2f}us (c_loop*{loop_trip})" if loop_trip > 1 else ""
    lines.append(
        f"  => R={R}B W={W}B: ({R}+{W})/{p.bw_peak_gbps:.0f} = {base / 1000:.2f}us "
        f"+ turnaround {turn / 1000:.2f}us (a*min(R,W)){extra} => T = {t / 1000:.2f} us"
    )
    return "\n".join(lines)
