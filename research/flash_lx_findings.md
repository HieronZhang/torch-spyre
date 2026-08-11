# Flash attention: CP-SAT is optimal, the default solver is 20–28 % off

**Retraction.** An earlier version of this document claimed *every* shipped LX allocator was
13–28 % off optimal on flash attention, CP-SAT included. **That was wrong.** CP-SAT's
allocation was reconstructed by hill climbing on its byte objective, and the climb was
suboptimal at that objective — it retained 8.3 MB where the exact optimum retains 10.4 MB.
Solved exactly, **CP-SAT finds the time-optimal allocation on every contested flash config**.
The finding that survives is about the *default* solver, not about CP-SAT.

**What holds.** On the two most contested flash configurations, `greedy` — the shipped default
— is **20.4 % and 28.3 %** slower than optimal, and `bestfit` 20.4 % and 22.2 %. CP-SAT and a
plain largest-first rule are both exactly optimal. This matches the softmax corpus
(`findings_lx.md` §5) rather than contradicting it, and the two together make the
recommendation to change the default solver considerably stronger.

**Status.** Predicted, not measured. §4 gives the runs that would check it.

```sh
python3 research/lx_experiment.py --exhaustive
```

---

## 1. Where the contention is

Seven flash bundles carry recorded features; three exceed the 1587 KB/core budget:

| config | measured | movable buffers | peak if all resident | over budget |
|---|---:|---:|---:|---:|
| `Lq=Lk=4096, htiles=8, qtiles=4` | 18,922 µs | 20 | 4096 K | **2.6×** |
| `Lq=Lk=2048, htiles=8, qtiles=1` | 11,131 µs | 20 | 3072 K | 1.9× |
| `Lq=Lk=2048, ktiles=2` | 12,601 µs | 20 | 1808 K | 1.1× |

"Movable" means produced *and* consumed inside the bundle, so the allocator may choose its
residency. Twenty movable buffers is 2²⁰ assignments — this is the workload where the
question is interesting, and softmax, with four, is not.

## 2. The result

Every objective solved exactly, by enumerating all feasible allocations:

| | keeps | predicted | regret |
|---|---:|---:|---:|
| **case 1 — `Lq=Lk=4096 h8 q4`**, 213,952 feasible allocations | | | |
| optimum | 16/20 | 669,365 µs | — |
| `cpsat` | 16/20 | 669,365 µs | **+0.0 %** |
| largest-first | 16/20 | 669,365 µs | +0.0 % |
| `bestfit` | 15/20 | 805,663 µs | **+20.4 %** |
| `greedy` *(default)* | 15/20 | 805,663 µs | **+20.4 %** |
| **case 2 — `Lq=Lk=2048 h8 q1`**, 345,600 feasible allocations | | | |
| optimum | 16/20 | 157,841 µs | — |
| `cpsat` | 16/20 | 157,841 µs | **+0.0 %** |
| largest-first | 16/20 | 157,841 µs | +0.0 % |
| `bestfit` | 16/20 | 192,893 µs | +22.2 % |
| `greedy` *(default)* | 17/20 | 202,556 µs | **+28.3 %** |
| **case 3 — `Lq=Lk=2048 k2`**, 992,256 feasible allocations | | | |
| optimum | 19/20 | 285,902 µs | — |
| `cpsat` / `bestfit` | 19/20 | 285,902 µs | +0.0 % |
| `greedy` *(default)* | 16/20 | 302,781 µs | +5.9 % |

Note case 2's `greedy`: it keeps **17** buffers and is the *slowest*. More residency is not
better residency — it fills the scratchpad with the wrong buffers and is forced to spill ones
that mattered more.

## 3. Why exhaustive search, and why nothing here should be approximated

Peak footprint is monotone in the allocation, so feasible allocations are **downward closed**:
if a set does not fit, no superset fits. A DFS pruning on first infeasibility visits only the
feasible family — 2·10⁵ to 10⁶ sets, 12–81 s per case — and every objective's argmax falls
out of the same walk. `fast_evaluator` makes that affordable by deserialising once and
flipping `ArgTraffic.mem` in place instead of deep-copying the bundle per evaluation.

**Hill climbing got this backwards twice, in opposite directions**, which is why
`exhaustive_policies` now solves every objective exactly:

- Climbing on **time** returned exactly CP-SAT's allocation and missed one 13 % faster. Regret
  measured against it read 0 % because the reference was too weak to see better.
- Climbing on **bytes** retained 8.3 MB against an exact optimum of 10.4 MB, making CP-SAT
  look 13–20 % off when it is exactly optimal. This produced the retracted claim above.

A third defect, found while chasing the first two: tie-breaking with `max()` over a *set* is
non-deterministic across processes, because Python randomises string hashing and flash has
five buffers at exactly 256 KB. The same config reported CP-SAT at 16 buffers in one process
and 15 in another. Fixed by ordering on `(size, name)`.

## 4. What would have to be measured

`LAYOUT_SOLVER` makes the compiler produce genuinely different allocations for the same
program, so the ranking is testable. Four runs per case, seconds each:

```sh
BENCH_OP=flash_attn FA_H=32 FA_LQ=4096 FA_LK=4096 FA_H_TILES=8 FA_LQ_TILES=4 \
  SENCORES=32 BENCH_REPS=7 LAYOUT_SOLVER=cpsat \
  python3 docs/source/user_guide/examples/profile_ops.py
```

…and the same with `greedy`, `firstfit`, `bestfit`. **The prediction to falsify: `cpsat`
beats `greedy` by ~20 % on case 1 and ~28 % on case 2.** That is far above the ~1 % noise
floor of a 7-repeat median, so the experiment can actually decide something.

## 5. Caveats

- **Predicted, not measured.** Nothing here has run on a device.
- **`bestfit` and `greedy` are still reconstructions**, though of deterministic sequential
  procedures rather than of an optimiser, so the failure mode above does not apply. Confirm
  with `SPYRE_DUMP_COST=1`, which reports which buffers actually landed in LX.
- **Absolute predictions are 15–45× high.** These features carry the pre-fix
  `tile_rows_per_core`, so 669,365 µs stands against a measured 18,922 µs. The ranking should
  survive, because the underfill and spill derates key on tile geometry rather than residency
  and so scale every allocation of a bundle equally — an argument, not a measurement.
- **This does not show a cost model beating CP-SAT.** It shows the opposite: on flash, as on
  softmax, CP-SAT's byte objective is exactly time-optimal. The case for a cost model in LX
  residency remains the narrow one in `findings_lx.md` §6.
