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

"""Score alternative LX allocations for a fused bundle, offline.

WHY THIS EXISTS. The compiler decides which intermediates live in LX scratchpad and which
spill to HBM, and today that decision carries no notion of time: the default solver is
program-order first-fit with no objective at all, and the only solver that optimises anything
(CP-SAT) minimises ``(read_count + is_intermediate) * size`` -- bytes, not microseconds. The
cost model does know about time, but it *consumes* the allocation rather than proposing one:
``dump_cost_model._mem_of_layout`` reads ``"lx" in layout.allocation`` into ``ArgTraffic.mem``,
and LX args are then excluded from ``read_bytes``/``write_bytes`` entirely.

So the question this module answers is the ranking one: **given several allocations that all
fit, does the cost model order them the way the hardware does?**

THE MODEL CANNOT CHOOSE ALONE. It has no capacity constraint. Asked to pick freely it would
put every buffer in LX, because LX traffic is priced at zero. Capacity is the allocator's
concern; ranking is the model's. This module therefore takes the feasible set as an input.

VALIDATION. ``--validate`` replays the one experiment that already exists: 56 configurations
in the recovered database were measured with ``LX_PLANNING=0`` and ``=1``. Flipping the
recorded features from lx to hbm should reproduce the measured effect of actually recompiling
with LX off. That is what makes the mutation defensible on programs we cannot measure.

    python3 research/lx_choice.py --validate       # mutation vs 56 measured on/off pairs
    python3 research/lx_choice.py --contested      # bundles where the choice matters
    python3 research/lx_choice.py --policies       # cost model vs the compiler's heuristics
    python3 research/lx_choice.py --monotonicity   # does more LX ever predict SLOWER?
    python3 research/lx_choice.py --enumerate --op add6
"""

import argparse
import copy
import itertools
import json
import os
import random
import statistics
import sys

import regex as re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools", "cost_model"))

import eval_model as E  # noqa: E402

cm = E.cm


#: The extractor names the SAME buffer differently on each side of a dependence: the producer
#: writes ``op3``, the consumer reads ``buf3``, and a coarse-tiling helper writes
#: ``coarse_tile_fill_buf3``. Matching producers to consumers by raw name therefore finds no
#: intermediates at all -- which is how this module first reported that a six-operand add
#: chain had nothing to allocate. Canonicalise to the trailing buffer index instead.
_BUF_ID = re.compile(r"^(?:op|buf)(\d+)$|_buf(\d+)$")


def _canon(name):
    """Buffer identity, independent of which side of the dependence names it."""
    m = _BUF_ID.search(name or "")
    if not m:
        return name
    return f"b{m[1] or m[2]}"


def bundle_intermediates(feats):
    """Canonical ids of buffers produced AND consumed inside the bundle.

    These are the only buffers an allocator may place in LX: a graph input has to be read
    from HBM and a graph output has to be written there, so neither is a free choice. Every
    other buffer is one an allocator could keep on chip, and is what an enumeration varies.
    """
    produced, consumed = set(), set()
    for op in feats:
        for a in op.get("args", []):
            side = produced if a.get("role") == "output" else consumed
            side.add(_canon(a.get("name")))
    return sorted(produced & consumed)


def apply_allocation(feats, in_lx):
    """A copy of ``feats`` with every buffer in ``in_lx`` marked resident, others spilled.

    Only the buffers in ``bundle_intermediates`` are touched; graph inputs and outputs keep
    whatever the extractor recorded, because their residency is not the allocator's to choose.
    """
    out = copy.deepcopy(feats)
    movable = set(bundle_intermediates(feats))
    for op in out:
        for a in op.get("args", []):
            cid = _canon(a.get("name"))
            if cid in movable:
                a["mem"] = "lx" if cid in in_lx else "hbm"
    return out


def predict(feats, params=None):
    """Predicted microseconds for a bundle given as serialized features."""
    return cm.predict_ops(_deserialize(feats), params or cm.CostParams()) / 1000.0


def _deserialize(feats):
    """Serialized feature dicts -> OpFeatures, mirroring eval_model's own path."""
    ops = []
    for f in feats:
        args = [
            cm.ArgTraffic(
                name=a.get("name", ""),
                role=a.get("role", "input"),
                mem=a.get("mem", "hbm"),
                elems=int(a.get("elems", 0)),
                broadcast=bool(a.get("broadcast", False)),
                loop_factor=int(a.get("loop_factor", 1) or 1),
                dims=list(a.get("dims") or []),
                logical=list(a.get("logical") or []),
            )
            for a in f.get("args", [])
        ]
        kw = {k: v for k, v in f.items() if k != "args"}
        kw["args"] = args
        ops.append(cm.OpFeatures(**kw))
    return ops


#: ``_lx_planning_size()`` at the default ``DXP_LX_FRAC_AVAIL=0.2``:
#: ``round_up_128((2 MiB - 64 KiB) * 0.8)``. See scratchpad/allocator.py:838-855, pinned by
#: tests/inductor/test_scratchpad_solver.py:70.
#:
#: This is 3.1x the cost model's ``lx_spill_cap_bytes`` (512 KB), which LOOKS like a bug in
#: the model and is not: setting that parameter to this value measurably degrades softmax
#: (RMS 21.1 -> 22.2 %). It is a fitted bandwidth-derate knee, not a capacity. Use this
#: constant for capacity questions and leave that one alone.
LX_CAPACITY_BYTES = 1_625_344


def buffer_footprints(feats):
    """Per-core bytes for each movable buffer, keyed by canonical id.

    The allocator budgets per core (``utils.py:150-158`` divides a buffer's device size by
    its writer's core count), so capacity questions are per-core questions.
    """
    movable = set(bundle_intermediates(feats))
    out = {}
    for op in feats:
        cores = max(1, int(op.get("cores") or 1))
        dt = int(op.get("dtype_bytes") or 2)
        for a in op.get("args", []):
            cid = _canon(a.get("name"))
            if cid in movable:
                out[cid] = max(out.get(cid, 0), int(a.get("elems", 0)) * dt // cores)
    return out


def buffer_lifetimes(feats):
    """``{canonical id: (first_use, last_use + 1)}`` over op index.

    Mirrors the allocator's own model: time is the position in the operation list
    (``scratchpad/utils.py:84``) and an interval runs from first to last use
    (``plan_solver.py:75-81``).
    """
    uses = {}
    for i, op in enumerate(feats):
        for a in op.get("args", []):
            uses.setdefault(_canon(a.get("name")), []).append(i)
    return {n: (min(v), max(v) + 1) for n, v in uses.items()}


def peak_footprint(feats, subset, fp=None, life=None, nt=None):
    """Largest per-core LX demand of ``subset`` at any single point in the bundle.

    SUM IS THE WRONG QUESTION. In a dependent chain the intermediates die as fast as they
    are born -- ``buf0`` is dead once ``buf1`` exists -- so a six-operand add holds about two
    tiles at a time, not four. Summing instead of peaking made this module reject an
    allocation the compiler had actually made, and under-predicted a measured 2.60x as 1.80x.

    ``fp``/``life``/``nt`` are precomputed by the enumerator; they are optional so a caller
    with a single subset does not have to build them.
    """
    fp = buffer_footprints(feats) if fp is None else fp
    life = buffer_lifetimes(feats) if life is None else life
    nt = len(feats) + 1 if nt is None else nt
    peak = 0
    for t in range(nt):
        live = sum(
            fp.get(n, 0) for n in subset if life.get(n, (0, 0))[0] <= t < life[n][1]
        )
        peak = max(peak, live)
    return peak


#: Enumeration is exponential in the number of movable buffers. Above this many, report the
#: bundle as too large to enumerate rather than silently sampling it -- a truncated search
#: that looks complete is worse than an admitted gap.
MAX_ENUMERATE = 16


def feasible_allocations(feats, capacity=LX_CAPACITY_BYTES):
    """Subsets of the movable buffers whose PEAK simultaneous per-core footprint fits.

    This is the allocator's half of the decision and the model has no opinion on it: LX
    traffic is priced at zero, so an unconstrained model would hold everything on chip.
    Enumerating without this filter is not a harmless approximation -- it is how this module
    first predicted a 2.5x LX benefit for four configurations the hardware measured at
    1.00x, because their single intermediate was 2048 KB against a 1587 KB budget and the
    compiler had already spilled it.
    """
    fp = buffer_footprints(feats)
    life = buffer_lifetimes(feats)
    nt = len(feats) + 1
    names = sorted(fp)
    if len(names) > MAX_ENUMERATE:
        raise ValueError(f"{len(names)} movable buffers exceeds MAX_ENUMERATE")
    out = []
    for r in range(len(names) + 1):
        for combo in itertools.combinations(names, r):
            if peak_footprint(feats, set(combo), fp, life, nt) <= capacity:
                out.append(frozenset(combo))
    return out


def rank_allocations(feats, feasible=None, params=None):
    """Score every feasible allocation, cheapest first.

    ``feasible`` is a list of frozensets of buffer names to hold in LX. Default: the full
    power set of the bundle's intermediates -- correct only when capacity is not binding,
    which is exactly why the caller normally supplies the feasible set instead.
    """
    movable = bundle_intermediates(feats)
    if feasible is None:
        feasible = [
            frozenset(c)
            for r in range(len(movable) + 1)
            for c in itertools.combinations(movable, r)
        ]
    scored = [(predict(apply_allocation(feats, s), params), s) for s in feasible]
    return sorted(scored, key=lambda t: t[0]), movable


def _pairs_from(records):
    """Configurations measured at BOTH LX_PLANNING=0 and =1."""
    import collections

    def key(r):
        return (
            r.get("op"),
            r.get("rows"),
            r.get("cols"),
            r.get("tiles"),
            r.get("cores"),
            (r.get("label") or "").split(" lx=")[0],
        )

    g = collections.defaultdict(dict)
    for r in records:
        if r.get("failed") or not r.get("kernel_us") or r.get("lx") not in (0, 1):
            continue
        g[key(r)].setdefault(r["lx"], r)
    return {k: v for k, v in g.items() if len(v) == 2}


def validate(records):
    """Does mutating lx->hbm reproduce the effect of recompiling with LX off?

    The comparison that matters is not the absolute time but the RATIO: recompiling with
    LX_PLANNING=0 slowed these kernels by a measured factor, and flipping the recorded
    features should predict that same factor. If it does, the mutation is a stand-in for a
    recompile, and allocations we cannot measure can still be ranked.
    """
    pairs = _pairs_from(records)
    rows = []
    for k, v in sorted(pairs.items()):
        on = v[1]
        if not on.get("feats"):
            continue
        meas = v[0]["kernel_us"] / v[1]["kernel_us"]
        movable = bundle_intermediates(on["feats"])
        if not movable:
            continue
        # LX_PLANNING=1 does not mean "everything on chip" -- it means "everything the
        # allocator could place". Ask the capacity filter, not the op list.
        best = max(feasible_allocations(on["feats"]), key=len)
        p_on = predict(apply_allocation(on["feats"], best))
        p_off = predict(apply_allocation(on["feats"], set()))
        rows.append((k, meas, p_off / p_on, len(best)))

    print(f"{len(rows)} configurations measured both ways, with features\n")
    print(
        f"{'op':<12} {'shape':<14} {'bufs':>4} {'measured':>9} {'mutated':>9} {'err':>8}"
    )
    ok = 0
    for k, meas, pred_ratio, n in sorted(rows, key=lambda r: -r[1]):
        err = (pred_ratio - meas) / meas * 100
        same = (meas > 1.05) == (pred_ratio > 1.05)
        ok += same
        if meas > 1.05 or abs(err) > 15:
            print(
                f"{k[0]:<12} {str(k[1]) + 'x' + str(k[2]):<14} {n:>4} "
                f"{meas:8.2f}x {pred_ratio:8.2f}x {err:+7.0f}%"
            )
    print(f"\ndirection agrees on {ok}/{len(rows)}")
    big = [r for r in rows if r[1] > 1.2]
    if big:
        errs = [abs((r[2] - r[1]) / r[1] * 100) for r in big]
        print(
            f"where LX matters (>1.2x, n={len(big)}): "
            f"median magnitude error {statistics.median(errs):.0f}%"
        )
    return rows


def contested(records, capacity=LX_CAPACITY_BYTES, min_spread=1.05):
    """Bundles where LX capacity binds AND the choice among feasible sets changes the time.

    That conjunction is the whole point. A bundle with room for everything has nothing to
    decide; a bundle where every feasible choice costs the same has nothing at stake. Only
    where both bite does an allocator's policy earn or lose anything, and only there is a
    cost model worth consulting.
    """
    seen, out = set(), []
    for r in records:
        feats = r.get("feats")
        if not feats or r.get("failed"):
            continue
        movable = bundle_intermediates(feats)
        if len(movable) < 2:
            continue
        key = (
            r.get("op"),
            r.get("rows"),
            r.get("cols"),
            r.get("tiles"),
            r.get("cores"),
        )
        if key in seen:
            continue
        seen.add(key)
        try:
            allocs = feasible_allocations(feats, capacity)
        except (
            ValueError
        ) as exc:  # too many buffers to enumerate; report, do not sample
            out.append((r, None, movable, float("nan")))
            print(f"SKIPPED {r.get('op')} {r.get('label')}: {exc}", file=sys.stderr)
            continue
        # Capacity binds iff holding every movable buffer is NOT feasible.
        if frozenset(movable) in allocs:
            continue
        ranked, _ = rank_allocations(feats, allocs)
        if len(ranked) < 2:
            continue
        spread = ranked[-1][0] / ranked[0][0]
        if spread >= min_spread:
            out.append((r, ranked, movable, spread))
    return sorted(out, key=lambda t: -t[3])


def monotonicity_violations(feats, capacity=LX_CAPACITY_BYTES):
    """Allocations where ADDING a buffer to LX makes the prediction SLOWER.

    This should be impossible. Moving a buffer from HBM to LX strictly removes HBM traffic
    and adds nothing, so predicted time must be non-increasing in the LX set. Where it is
    not, the model is not internally consistent across allocations -- and a model that is
    used to *rank* allocations has to be.

    The known cause is a branch discontinuity in ``predict_ops``. Spilling a small
    broadcast operand (e.g. softmax's ``amax`` column) makes its consumer a broadcast op,
    which flips the whole bundle from the single-bandwidth + ``alpha*min(R,W)`` turnaround
    model to the per-op effective-bandwidth model -- and the latter folds turnaround into
    the rate rather than charging it separately. The formula changes, so the prediction can
    fall even though the traffic rose.

    Returns (subset, superset, us_subset, us_superset) for each violation.
    """
    allocs = feasible_allocations(feats, capacity)
    us = {s: predict(apply_allocation(feats, s)) for s in allocs}
    out = []
    for small in allocs:
        for big in allocs:
            if small < big and us[big] > us[small] * 1.0005:
                out.append((small, big, us[small], us[big]))
    return out


def fast_evaluator(feats, params=None):
    """A ``subset -> predicted us`` closure that avoids re-copying the features.

    ``apply_allocation`` deep-copies the whole bundle on every call, which is fine for a
    handful of allocations and hopeless for an exhaustive search -- enumerating flash
    attention's feasible set that way did not finish in two minutes. Deserialise once, keep
    references to the movable args, and flip ``mem`` in place instead. Same arithmetic,
    roughly two orders of magnitude faster.
    """
    p = params or cm.CostParams()
    ops = _deserialize(feats)
    movable = set(bundle_intermediates(feats))
    refs = [
        (a, _canon(a.name)) for op in ops for a in op.args if _canon(a.name) in movable
    ]
    cache = {}

    def evaluate(subset):
        key = frozenset(subset)
        hit = cache.get(key)
        if hit is None:
            for arg, cid in refs:
                arg.mem = "lx" if cid in key else "hbm"
            hit = cache[key] = cm.predict_ops(ops, p) / 1000.0
        return hit

    return evaluate


def exhaustive_best(feats, capacity=LX_CAPACITY_BYTES, budget=2_000_000):
    """The provably optimal feasible allocation, by exhaustive search with pruning.

    Peak footprint is monotone in the subset -- adding a buffer can only raise it -- so the
    feasible allocations are downward closed and a DFS can prune every superset of an
    infeasible set. That makes 20 buffers tractable where 2^20 blind evaluations are not.

    Returns ``(best_subset, best_us, n_evaluated, complete)``. ``complete`` is False if the
    evaluation budget was hit, in which case the answer is a strong upper bound rather than
    a proven optimum -- and says so rather than pretending otherwise.
    """
    fp = buffer_footprints(feats)
    life = buffer_lifetimes(feats)
    nt = len(feats) + 1
    names = sorted(fp, key=lambda n: (-fp[n], n))
    evaluate = fast_evaluator(feats)
    best, best_us, seen, complete = frozenset(), float("inf"), 0, True

    def rec(i, cur, peak_cache):
        nonlocal best, best_us, seen, complete
        if not complete:
            return
        if i == len(names):
            seen += 1
            if seen > budget:
                complete = False
                return
            us = evaluate(cur)
            if us < best_us:
                best, best_us = frozenset(cur), us
            return
        rec(i + 1, cur, peak_cache)  # exclude
        cand = cur | {names[i]}
        if peak_footprint(feats, cand, fp, life, nt) <= capacity:
            rec(i + 1, cand, peak_cache)  # include, only while it still fits

    rec(0, set(), {})
    return best, best_us, seen, complete


def exhaustive_policies(feats, capacity=LX_CAPACITY_BYTES):
    """One enumeration, every objective solved EXACTLY.

    NEVER APPROXIMATE A SOLVER'S OWN OBJECTIVE. Hill climbing was used for this and got the
    answer backwards twice, in opposite directions:

    * climbing on TIME returned exactly CP-SAT's allocation and missed one 13 % faster, so
      CP-SAT's regret read 0 % when the search was simply too weak to see better;
    * climbing on BYTES retained 8.3 MB where the exact byte-optimum retains 10.4 MB, so
      CP-SAT looked 13-20 % off optimal when, solved properly, it is exactly optimal.

    The second error produced a whole document claiming every shipped allocator was 13-28 %
    off. It was wrong. Feasible allocations are downward closed, one DFS visits them all, and
    each objective's argmax falls out of the same walk -- so there is no reason to approximate.

    Returns ``{name: subset}`` for ``time`` (the true optimum), ``cpsat`` (max retained
    ``(reads+1)*size``), ``largest`` (max retained bytes), plus the evaluator and the count.
    """
    fp = buffer_footprints(feats)
    life = buffer_lifetimes(feats)
    nt = len(feats) + 1
    rc = read_counts(feats)
    evaluate = fast_evaluator(feats)
    names = sorted(fp, key=lambda n: (-fp[n], n))
    best = {
        "time": [None, float("inf"), lambda s: evaluate(s)],
        "cpsat": [None, -1.0, lambda s: sum((rc.get(x, 0) + 1) * fp[x] for x in s)],
        "largest": [None, -1.0, lambda s: sum(fp[x] for x in s)],
    }
    count = 0

    def visit(cur):
        nonlocal count
        count += 1
        frozen = frozenset(cur)
        for key, slot in best.items():
            v = slot[2](frozen)
            better = v < slot[1] if key == "time" else v > slot[1]
            if better:
                slot[0], slot[1] = frozen, v

    def rec(i, cur):
        if i == len(names):
            visit(cur)
            return
        rec(i + 1, cur)
        cand = cur | {names[i]}
        if peak_footprint(feats, cand, fp, life, nt) <= capacity:
            rec(i + 1, cand)

    rec(0, set())
    return {k: v[0] for k, v in best.items()}, evaluate, count


def search_best(feats, capacity=LX_CAPACITY_BYTES, restarts=12, seed=0, seeds=()):
    """Best feasible allocation found by hill climbing, for bundles too big to enumerate.

    WHY THIS EXISTS. Flash attention has 20 movable buffers -- 2^20 allocations -- so the
    exhaustive path refuses it. Refusing was the right call, but the skip was silent enough
    that an earlier version of this study reported "37 contested bundles, all softmax" while
    quietly passing over the three flash configurations that exceed capacity by up to 2.6x.
    Those are the most contested bundles in the corpus.

    Climbs on single add/remove moves from several starts, keeping the best feasible point
    seen. The result is a strong reference, NOT a proven optimum, so a policy's regret
    against it is a LOWER bound on its true regret.

    PASS THE POLICIES' OWN PICKS AS ``seeds``. Single-toggle climbing plateaus on this
    landscape -- the model is not monotone in residency (see ``monotonicity_violations``), so
    local optima are real. A first version of this search omitted the policy picks and duly
    reported CP-SAT beating the "best found" reference by 11.5%, which is impossible and was
    the signal that the search, not CP-SAT, was at fault. Seeding guarantees the reference
    dominates every policy it is used to score.
    """
    rng = random.Random(seed)
    fp = buffer_footprints(feats)
    life = buffer_lifetimes(feats)
    nt = len(feats) + 1
    names = sorted(fp)
    cache = {}

    def ok(s):
        return peak_footprint(feats, s, fp, life, nt) <= capacity

    def cost(s):
        key = frozenset(s)
        if key not in cache:
            cache[key] = (
                predict(apply_allocation(feats, key)) if ok(key) else float("inf")
            )
        return cache[key]

    def climb(start):
        cur = set(start)
        while not ok(cur) and cur:  # repair: drop the largest until it fits
            cur.discard(max(sorted(cur), key=lambda n: (fp[n], n)))
        best, bestc = set(cur), cost(cur)
        improved = True
        while improved:
            improved = False
            for n in names:
                cand = best ^ {n}
                if ok(cand) and cost(cand) < bestc - 1e-9:
                    best, bestc, improved = set(cand), cost(cand), True
        return frozenset(best), bestc

    starts = [
        set(),
        set(names),
        set(sorted(names, key=lambda n: -fp[n])[: len(names) // 2]),
    ]
    starts += [set(s) for s in seeds]
    starts += [
        {n for n in names if rng.random() < 0.5} for _ in range(max(0, restarts - 3))
    ]
    results = [climb(s) for s in starts]
    return min(results, key=lambda t: t[1])


def read_counts(feats):
    """How many ops read each buffer -- the quantity CP-SAT's spill cost is built from."""
    n = {}
    for op in feats:
        for a in op.get("args", []):
            if a.get("role") != "output":
                cid = _canon(a.get("name"))
                n[cid] = n.get(cid, 0) + 1
    return n


def policy_choices(feats, allocs):
    """What each existing allocator policy would pick, from the same feasible set.

    Reimplemented from the source so the comparison is against what the compiler actually
    does, not an idealisation:

    * ``cpsat``  -- maximise retained ``(read_count + 1) * size``, i.e. minimise the
      differential HBM bytes of what spills (``ilp_solver_ortools.py:208-224, 467``).
    * ``largest`` -- a size-first bin-packer, the obvious baseline.
    * ``program_order`` -- take buffers in first-use order while they still fit, which is
      what the DEFAULT greedy solver's chronological bump allocation amounts to
      (``greedy_solver.py:189-198``).
    """
    fp = buffer_footprints(feats)
    rc = read_counts(feats)
    allocs = list(allocs)

    def value(s):
        return sum((rc.get(n, 0) + 1) * fp[n] for n in s)

    cpsat = max(allocs, key=value)
    largest = max(allocs, key=lambda s: (sum(fp[n] for n in s), len(s)))
    life = buffer_lifetimes(feats)
    allowed = set(allocs)

    def _take_in_order(order):
        """Place buffers in `order`, keeping each one that still fits."""
        chosen = []
        for n in order:
            if frozenset(chosen + [n]) in allowed:
                chosen.append(n)
        return frozenset(chosen)

    # `greedy` (the DEFAULT): chronological bump allocation, so buffers are reached in
    # first-use order and whoever arrives when LX is full loses (greedy_solver.py:189-198).
    program_order = _take_in_order(
        sorted(fp, key=lambda n: (life.get(n, (0, 0))[0], n))
    )

    # `firstfit` / `bestfit`: both order by (lifetime - discount) / uses ascending, so
    # short-lived heavily-reused buffers are placed first (firstfit_bestfit_solver.py:
    # 204-216). They differ only in WHICH free gap they pick, which does not change the
    # residency set, so one ordering stands for both here. `in_place_parents` is not
    # recoverable from recorded features, so the 0.25 discounts are omitted; and an
    # intermediate's first use is its write, so `first_use_is_read` is False and the
    # denominator gains the +0.5 write bonus.
    def _fit_key(n):
        start, end = life.get(n, (0, 0))
        uses = rc.get(n, 0) + 1  # the producing write plus each consuming read
        return ((end - start) / (uses + 0.5), end - start, n)

    fit_order = _take_in_order(sorted(fp, key=_fit_key))

    return {
        "cpsat": cpsat,
        "largest": largest,
        "program_order": program_order,
        "firstfit/bestfit": fit_order,
    }


def compare_policies(records, capacity=LX_CAPACITY_BYTES):
    """Where does time-based ranking disagree with the byte-based heuristics in the tree?

    This is the question that decides whether a cost model is worth wiring into the
    allocator at all. If the existing byte proxies already pick the time-optimal allocation
    everywhere, the model adds nothing here and the honest answer is to say so.
    """
    rows = []
    for r, ranked, movable, spread in contested(records, capacity):
        if ranked is None:
            continue
        feats = r["feats"]
        allocs = [s for _, s in ranked]
        best_us, best = ranked[0]
        by_us = {s: us for us, s in ranked}
        picks = policy_choices(feats, allocs)
        rows.append(
            (
                r,
                best_us,
                best,
                {k: (by_us[v], v) for k, v in picks.items()},
                spread,
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--enumerate", dest="enum", action="store_true")
    ap.add_argument("--contested", action="store_true")
    ap.add_argument("--policies", action="store_true")
    ap.add_argument("--monotonicity", action="store_true")
    ap.add_argument("--capacity", type=int, default=LX_CAPACITY_BYTES)
    ap.add_argument("--op", default="")
    args = ap.parse_args()

    path = args.records or E.records_path()
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)["records"]

    if args.validate:
        validate(records)
        return 0

    if args.monotonicity:
        hits = contested(records, args.capacity)
        bad = tot = 0
        print("auditing: does adding a buffer to LX ever make the prediction SLOWER?\n")
        for r, ranked, movable, spread in hits:
            if ranked is None:
                continue
            tot += 1
            v = monotonicity_violations(r["feats"], args.capacity)
            if v:
                bad += 1
                if bad <= 4:
                    small, big, a, b = max(v, key=lambda t: t[3] - t[2])
                    print(f"{(r.get('label') or '')[:58]}")
                    print(f"   LX={sorted(small)} -> {a:8.1f} us")
                    print(
                        f"   LX={sorted(big)} -> {b:8.1f} us   (+{(b - a) / a * 100:.1f}% for MORE residency)"
                    )
                    print(f"   {len(v)} violating pairs in this bundle\n")
        print(f"{bad}/{tot} contested bundles contain a monotonicity violation")
        return 0

    if args.policies:
        rows = compare_policies(records, args.capacity)
        print(
            f"{len(rows)} contested bundles. Regret = how much slower a policy's pick is"
        )
        print("than the time-optimal feasible allocation, in predicted us.\n")
        pols = ("cpsat", "largest", "firstfit/bestfit", "program_order")
        print(f"{'op':<20} {'best us':>9} " + " ".join(f"{k:>17}" for k in pols))
        agree = {k: 0 for k in pols}
        worst = {k: 0.0 for k in agree}
        for r, best_us, best, picks, spread in rows:
            cells = []
            for k in pols:
                us, _ = picks[k]
                reg = (us - best_us) / best_us * 100
                agree[k] += reg < 0.5
                worst[k] = max(worst[k], reg)
                cells.append(f"{reg:+15.1f}%")
            print(f"{(r.get('op') or '')[:20]:<20} {best_us:9.1f} " + " ".join(cells))
        print()
        for k in pols:
            print(
                f"  {k:<14} optimal on {agree[k]}/{len(rows)}, worst regret {worst[k]:+.1f}%"
            )
        return 0

    if args.contested:
        hits = contested(records, args.capacity)
        print(
            f"LX budget {args.capacity / 1024:.0f} KB/core; "
            f"{len(hits)} bundles where capacity binds and the choice matters\n"
        )
        for r, ranked, movable, spread in hits[:12]:
            fp = buffer_footprints(r["feats"])
            print(f"{r.get('op')}  {r.get('label')}")
            print(
                f"   {len(movable)} movable: "
                + ", ".join(f"{n}={fp[n] / 1024:.0f}K" for n in movable)
                + f"   peak-if-all={peak_footprint(r['feats'], set(movable)) / 1024:.0f}K"
            )
            print(
                f"   best  {ranked[0][0]:8.1f} us  LX={sorted(ranked[0][1]) or '(none)'}"
            )
            print(
                f"   worst {ranked[-1][0]:8.1f} us  LX={sorted(ranked[-1][1]) or '(none)'}"
            )
            print(f"   {len(ranked)} feasible choices, spread {spread:.2f}x\n")
        return 0

    if args.enum:
        cands = [
            r
            for r in records
            if r.get("feats") and (not args.op or r.get("op") == args.op)
        ]
        if not cands:
            sys.exit(f"no records with features for op={args.op!r}")
        r = cands[0]
        ranked, movable = rank_allocations(r["feats"])
        print(f"{r.get('label')}   measured {r['kernel_us']:.1f} us")
        print(f"movable intermediates: {movable}\n")
        for us, s in ranked:
            print(f"  {us:9.1f} us   LX = {sorted(s) or '(none)'}")
        print(
            f"\nspread {ranked[-1][0] / ranked[0][0]:.2f}x across {len(ranked)} choices"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
