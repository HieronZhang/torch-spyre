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

    T = fill + [ (R + W) / BW_PEAK + alpha * min(R, W) ] / eff_underfill
            + combine + c_loop * L

  where R = HBM bytes READ (inputs), W = HBM bytes WRITTEN (outputs). LX-resident
  traffic is treated as ~free. ``eff_underfill`` (<=1) derates the bandwidth term for
  OUTPUT-dim (pointwise) coarse-tiling that shrinks each core's per-tile height (see
  the coarse-tiling bullet below); ``combine`` is the reduction ring term; and
  ``c_loop * L`` is the per-iteration coarse-tiling loop overhead (L = trip count). The
  derate and c_loop*L are DISTINCT mechanisms -- a per-tile-SIZE throughput loss vs a
  per-ITERATION fixed cost -- so both can apply to one tiled loop. For a normal untiled
  kernel eff_underfill = 1, combine = 0, L = 1, so this reduces to the bandwidth model.

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
is split across cores.

MATMUL (reduction_type batchmatmul) is COMPUTE-bound, so it gets an extra ADDITIVE term
``compute = MACs / cores / (mac_peak * pt_eff)`` (MACs = M*N*K). ``pt_eff`` reuses the
underfill derate (target_passes=8; matmul saturates ~64 rows/core). The K-split ring
reduction reuses the ``combine`` term (reduction_cores = k). VALIDATED on the mm K-sweep
(M=N=2048, K 512..8192): measured = compute_datasheet + the bandwidth turnaround term to
~1% for K>=1024 -- ADDITIVE (no compute/HBM overlap) and the datasheet MAC peak is
correct; the only fix vs the in-tree Pass-2 model (work_division.py) is its 204.8 GB/s
-> our turnaround BW. See notes/cost_model_summary.md.

COARSE-TILING UNDERFILL: tiling an OUTPUT dim (a fused pointwise chain split so an
intermediate stays in LX) cuts each core's per-tile height. Total HBM bytes are
unchanged, but a short per-core tile underfills the streaming pipeline (fill/drain not
amortised), derating effective throughput. We model it as ``eff_underfill =
min(1, (rows_per_core / r_full) ** exp)`` -- the SAME shape as the matmul ``pt_eff``
(work_division.py); the shared hardware constant is the 8-row pass, only the saturation
point differs by op structure (pointwise saturates ~16 rows/core, matmul ~64). This is
ADDED on top of the per-iteration ``c_loop * L`` (a different mechanism, see below), not
a replacement: the chain K-sweep ([2048,4096], LX on) is flat to ~16 rows/core then
cliffs (8 rows/core ~+34%, 4 rows/core ~+53%) -- a flat-then-cliff shape that a linear
c_loop*L cannot produce (and at 860 ns/tile c_loop adds only ~14us at 16 tiles vs the
~+300us observed), so the underfill derate is the DOMINANT pointwise-tiling term while
c_loop*L stays as the small loop-dispatch cost. (In the chain sweep L and rows/core are
anti-correlated 64/L, so the two are confounded -- the untiled-small-ROWS confirm runs,
L=1 with short tiles, isolate the underfill so the pointwise c_loop can be pinned
separately.) PROVISIONAL: r_full and exp are guessed from the chain sweep.

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
    # LOGICAL torch shape, e.g. [512, 1024] -- shown next to dims so the stickification
    # (a row of N rounds up to ceil(N/64)*64 sticks) is visible per tensor.
    logical: list = dataclasses.field(default_factory=list)
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
    # OUTPUT-dim (pointwise) coarse-tiling: True when this op tiles an output dim, so
    # each core's per-tile height shrinks and the underfill derate applies. False for
    # reduction-dim tiling (keeps the calibrated c_loop) and for untiled ops.
    tiles_output_dim: bool = False
    # Per-core per-tile pass-row height (output-tiled ops only): the streamed tile's
    # "rows" / cores. Drives ``eff_underfill``; 0.0 = unknown / not applicable -> no
    # derate.
    tile_rows_per_core: float = 0.0
    # MATMUL (reduction_type batchmatmul): adds an ADDITIVE compute term. matmul_macs =
    # M*N*K (total multiply-accumulates); matmul_rows_per_core = M/m (per-core M tile,
    # drives pt_eff). K-split k is carried in ``reduction_cores`` (-> the combine/PSUM
    # term). All zero/False for non-matmul ops.
    is_matmul: bool = False
    matmul_macs: int = 0
    matmul_rows_per_core: float = 0.0

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
    # REDUCTION-DIM coarse-tiling per-iteration loop overhead (ns/iteration). CALIBRATED
    # ~860 ns/tile from the P2a K-sweep (ctsum B=2048 D=512, tiles 2..16): unmodeled =
    # (kernel - pred) fits 1.88us + 0.864*K, so the K-slope is the loop overhead (the
    # 1.88us intercept is the dim0-reduction access penalty -- a separate sumcol-like
    # bias, NOT loop cost; the D-sweep gives the same K-slope, so it is data-indep).
    # This is the per-ITERATION loop-dispatch cost and applies to EVERY tiled loop
    # (pointwise + reduction). It is a DIFFERENT mechanism from the underfill derate
    # below (per-tile-SIZE); both can apply. For a tiled reduction c_loop*L is the main
    # tiling term; for a pointwise chain it is the small term (underfill dominates). The
    # 860 ns is calibrated from the reduction K-sweep; the pointwise value is confounded
    # with underfill in current data (pending the untiled-small-ROWS confirm runs).
    c_loop_ns: float = 860.0
    # Pipeline-fill (underfill) derate for OUTPUT-dim (pointwise) coarse-tiling:
    # eff = min(1, (rows_per_core / (pass_rows * target_passes)) ** exponent). Same FORM
    # as the matmul pt_eff (work_division.py); the 8-row pass is the shared hardware
    # constant, target_passes differs by op structure. PROVISIONAL -- guessed from the
    # chain K-sweep (flat to ~16 rows/core, cliff at 8; data hints exponent ~0.4). To be
    # calibrated by the untiled-small-ROWS underfill-confirm runs.
    underfill_pass_rows: float = 8.0  # PT / stream pass granularity (matmul _PT_ROWS)
    underfill_target_passes_pointwise: float = 2.0  # pointwise full-fill ~2 pass (=16)
    underfill_exponent: float = 0.5  # falloff (sqrt, as matmul pt_eff; chain data ~0.4)
    # MATMUL compute term (ADDITIVE with the bandwidth term -- no compute/HBM overlap).
    # T_matmul = compute + H_turnaround + psum, compute = MACs/cores/(mac_peak*pt_eff).
    # mac_peak = datasheet (98.304e12/2/32 MAC/us/core = 1536 MAC/ns/core) -- VALIDATED
    # by the mm K-sweep (M=N=2048, K 512..8192): measured = compute_datasheet +
    # H_turnaround to ~1% for K>=1024, so the peak is right and the only fix vs the
    # in-tree Pass-2 model is its 204.8 GB/s -> our turnaround BW. pt_eff reuses the
    # underfill derate with target_passes=8 (matmul saturates ~64 rows/core).
    mac_peak_per_core_ns: float = 1536.0  # MAC/ns/core (datasheet; K-sweep checked)
    underfill_target_passes_matmul: float = 8.0  # matmul full-fill ~8 passes (=64 rows)


def underfill_eff(
    rows_per_core: float,
    params: CostParams | None = None,
    target_passes: float | None = None,
) -> float:
    """Pipeline-fill efficiency (<=1) for a per-core tile of ``rows_per_core`` rows.

    The streaming / PT pipeline processes a core's tile in passes of
    ``underfill_pass_rows`` (8) rows; a tile shorter than ``pass_rows * target_passes``
    cannot amortise pipeline fill/drain, so effective throughput derates as
    ``(rows / r_full) ** exponent``, capped at 1. ``target_passes`` defaults to the
    pointwise value (coarse-tiling); pass ``underfill_target_passes_matmul`` for the
    matmul compute term (same FORM, deeper pipeline). ``rows_per_core <= 0`` (unknown)
    -> 1.0 (no derate).
    """
    p = params or CostParams()
    if rows_per_core <= 0:
        return 1.0
    tp = p.underfill_target_passes_pointwise if target_passes is None else target_passes
    r_full = p.underfill_pass_rows * tp
    if r_full <= 0:
        return 1.0
    return min(1.0, (rows_per_core / r_full) ** p.underfill_exponent)


def predict_ops(ops: list, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a bundle of ops (one fused kernel).

    ``T = fill + [(R+W)/BW_PEAK + alpha*min(R,W)] / eff_underfill + combine +
    c_loop*L_red`` where R/W are the bundle's total HBM read/write bytes (LX ~free),
    already loop-scaled per arg (see ArgTraffic.loop_factor). R and W are summed over
    the whole bundle before the turnaround term, since a fused kernel interleaves all
    its reads and writes on a shared bus. ``eff_underfill`` derates the bandwidth term
    when OUTPUT-dim (pointwise) tiling shortens each core's per-tile height. Reductions
    add a ring-combine term (once PER TILE); reduction-dim tiling adds c_loop*L_red.
    Matmul ops add an ADDITIVE compute term (MACs/cores/(mac_peak*pt_eff)).
    """
    p = params or CostParams()
    r = sum(o.read_bytes() for o in ops)
    w = sum(o.write_bytes() for o in ops)
    mem = (r + w) / p.bw_peak_gbps + p.rw_turnaround_ns_per_byte * min(r, w)
    # OUTPUT-dim (pointwise) coarse-tiling underfill: a short per-core tile underfills
    # the streaming pipeline, derating the bandwidth term. The smallest tile in the
    # bundle governs (worst underfill). 1.0 (no derate) when nothing is output-tiled.
    eff = 1.0
    for o in ops:
        if o.loop_trip > 1 and o.tiles_output_dim and o.tile_rows_per_core > 0:
            eff = min(eff, underfill_eff(o.tile_rows_per_core, p))
    t = p.fill_ns + mem / eff
    # MATMUL compute (ADDITIVE with the bandwidth term -- the mm K-sweep showed no
    # compute/HBM overlap). compute = MACs/cores derated by pt_eff (PT-array fill).
    for o in ops:
        if o.is_matmul and o.matmul_macs > 0 and o.cores > 0:
            pt_eff = underfill_eff(
                o.matmul_rows_per_core, p, p.underfill_target_passes_matmul
            )
            t += o.matmul_macs / o.cores / (p.mac_peak_per_core_ns * pt_eff)
    for o in ops:
        if o.is_reduction:
            combine = max(0, o.reduction_cores - 1) * o.out_elems * p.psum_per_elem_ns
            t += combine * o.loop_trip
    # Coarse-tiling per-iteration LOOP overhead: each tiled-loop iteration pays a fixed
    # dispatch/setup cost (calibrated c_loop, linear in the trip count L) -- a DIFFERENT
    # mechanism from the underfill derate above (a per-tile-SIZE throughput derate), so
    # both can apply. For a pointwise chain the underfill dominates and c_loop*L is the
    # small term; for a tiled reduction underfill ~= 1 and c_loop*L is the main tiling
    # term. L = max loop trip over the bundle.
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
            dev = a.dims if a.dims else [a.elems]
            log = f"torch {a.logical} -> " if a.logical else ""
            # One line per DEVICE-LAYOUT tensor: name, role, logical->device dims,
            # residency, byte calc, the HBM bytes the model counts, and the loop factor.
            lines.append(
                f"      {a.role:<6} {a.name:<22} {log}device {dev} in {a.mem.upper()}"
                f"  | {a.elems} elems x {o.dtype_bytes}B = {a.elems * o.dtype_bytes} B"
                f" (hbm counted: {counted} B){lf}{bc}"
            )
    # Prediction with the rough calculation spelled out, so SPYRE_DUMP_COST shows the
    # same step-by-step breakdown (base + turnaround, then the underfill derate for
    # pointwise tiling or the c_loop term for reduction-dim tiling).
    R = sum(o.read_bytes() for o in ops)
    W = sum(o.write_bytes() for o in ops)
    base = (R + W) / p.bw_peak_gbps
    turn = p.rw_turnaround_ns_per_byte * min(R, W)
    # Underfill derate (output-dim tiling): smallest per-core tile governs.
    eff, eff_rows = 1.0, 0.0
    for o in ops:
        if o.loop_trip > 1 and o.tiles_output_dim and o.tile_rows_per_core > 0:
            e = underfill_eff(o.tile_rows_per_core, p)
            if e < eff:
                eff, eff_rows = e, o.tile_rows_per_core
    loop_trip = max((o.loop_trip for o in ops), default=1)
    # Matmul compute (additive): sum the per-op compute term for any matmul ops.
    mm_us, mm_lines = 0.0, []
    for o in ops:
        if o.is_matmul and o.matmul_macs > 0 and o.cores > 0:
            pe = underfill_eff(
                o.matmul_rows_per_core, p, p.underfill_target_passes_matmul
            )
            c_ns = o.matmul_macs / o.cores / (p.mac_peak_per_core_ns * pe)
            mm_us += c_ns / 1000
            mm_lines.append(
                f"     compute = MACs/cores/(mac_peak*pt_eff) = {o.matmul_macs}/"
                f"{o.cores}/({p.mac_peak_per_core_ns:.0f}*{pe:.3f}) = {c_ns / 1000:.2f}"
                f" us  (M/m={o.matmul_rows_per_core:.0f}, pt_eff={pe:.3f})"
            )
    t = predict_ops(ops, p)
    parts = "(R+W)/BW_PEAK + a*min(R,W)"
    if eff < 1.0:
        parts = f"[{parts}] / eff_underfill"
    if mm_us > 0:
        parts = f"compute + {parts}"
    if loop_trip > 1:
        parts += " + c_loop*L"
    lines.append(f"  -- prediction (turnaround): T = {parts} --")
    lines.append(f"     R={R}B (read)   W={W}B (write)")
    lines.extend(mm_lines)
    lines.append(f"     base = (R+W)/BW_PEAK = ({R}+{W})/{p.bw_peak_gbps:.0f} "
                 f"= {base / 1000:.2f} us")
    lines.append(f"     turn = a*min(R,W) = {p.rw_turnaround_ns_per_byte}*{min(R, W)} "
                 f"= {turn / 1000:.2f} us")
    if eff < 1.0:
        rf = p.underfill_pass_rows * p.underfill_target_passes_pointwise
        lines.append(f"     eff_underfill = min(1,({eff_rows:.1f}/{rf:.0f})"
                     f"**{p.underfill_exponent}) = {eff:.3f}  "
                     f"-> (base+turn)/eff = {(base + turn) / eff / 1000:.2f} us")
    if loop_trip > 1:
        loop_us = p.c_loop_ns * loop_trip / 1000
        lines.append(f"     loop = c_loop*L = {p.c_loop_ns:.0f}*{loop_trip} "
                     f"= {loop_us:.2f} us")
    lines.append(f"     => T_model = {t / 1000:.2f} us")
    return "\n".join(lines)
