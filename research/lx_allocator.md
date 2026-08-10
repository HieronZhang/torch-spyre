# How LX scratchpad allocation works today

A source-level reading of `torch_spyre/_inductor/scratchpad/`, written to answer one
question: **where, if anywhere, could a runtime cost model change what the allocator
decides?** Every claim below was checked against the code and, where one exists, the test
that pins it.

Companion: `research/findings_lx.md` (what the decision is worth, measured).

---

## 1. Where it runs

`scratchpad_planning()` (`allocator.py:2193`) is the **last** pass in
`CustomPreSchedulingPasses` (`passes.py:392-398`, gated on `config.lx_planning`). It runs
after coarse tiling and work division, so it sizes buffers against an already-reduced
per-tile, per-core working set — and *before* the Inductor Scheduler exists, so it sees a
flat `graph.operations` list rather than fusion bundles.

Two later passes can still undo its work:

- `scheduler.py:402` `demote_incoherent_lx_buffers` — a post-fusion correctness backstop that
  pops `"lx"` when a producer and consumer disagree on loop order.
- `hbm_pool_planning.py:237-246` — claims everything LX did *not* take.

The cost model (`cost_model_pass`, `passes.py:508`) runs **after** allocation. It can score a
finished plan; it cannot steer one.

## 2. The decision is made in two layers, and neither knows about time

| layer | who decides | on what basis |
|---|---|---|
| **eligibility** — may this buffer be in LX at all? | `ScratchpadAllocator._residency_reasons` (`allocator.py:401`) | legality + a hard-coded op allowlist |
| **placement** — where in LX, and does it fit? | `MemoryPlanSolver.plan_layout` (`plan_solver.py:260`) | one of five solvers |

`plan_layout` returns the buffers with `address: Optional[int]` set, and **`address is None`
*is* the decision to live in HBM**. That single field is the entire output of LX planning.

### Eligibility: a 19-op allowlist

`allocator.py:215-226`:

```python
return config.allow_all_ops_in_lx_planning or (
    self._get_op_name(op) in OP_OUTPUT_GOOD_FOR_LX_REUSE
)
```

`OP_OUTPUT_GOOD_FOR_LX_REUSE` (`utils.py:43`) is exactly:

```
max amax maximum sum clone exp sub mul mean add rsqrt neg
mm bmm batched_matmul div realdiv expand silu
```

Two consequences worth noting, because they are easy to get wrong:

- **This is why softmax dominates the contested cases.** Its decomposition — `amax`, `sub`,
  `exp`, `sum`, `div` — is entirely on the list, so every one of its intermediates competes
  for LX.
- **Transports are *not* excluded, despite `transpose`/`cat` being absent.** A materialised
  transpose or concat lowers to `clone`, which *is* on the list. In the recorded corpus
  `clone` carries `stick_scatter` 46 times and `restickify` 42 times. This matters: those are
  precisely the buffers whose effective bandwidth is far below default, and they are
  LX-eligible.

## 3. Capacity: 1,625,344 bytes per core

`_lx_planning_size()` (`allocator.py:838-855`):

```
round_up_128( (2 MiB − 64 KiB) × (1 − DXP_LX_FRAC_AVAIL) )
```

At the default `DXP_LX_FRAC_AVAIL = 0.2` that is **1,625,344 B ≈ 1.55 MiB/core**, pinned by
`tests/inductor/test_scratchpad_solver.py:70` (`(0.0, 2_031_616), (0.2, 1_625_344),
(1.0, 0)`).

Read the docstring carefully: this is an **ownership boundary**, not a safety margin. The
frontend reserves `1 − DXP_LX_FRAC_AVAIL` from address zero and DeepTools allocates at or
above it. Raising `DXP_LX_FRAC_AVAIL` does not "free up" LX — it hands it to the backend.

Capacity is per core because footprints are per core: `utils.py:150-158` divides a buffer's
device size by its writer's core count.

**Do not confuse this with `MAX_SPAN_BYTES`** (`work_division.py:73`) = 65535 × 4096 =
255.996 MiB, which is the HBM per-core *address span* limit. Unrelated to LX, and the source
of the `[4,1024,1024,1024]` CRITICAL in the 2026-08-07 sweep log.

**Not a model bug — a naming collision.** The cost model's `lx_spill_cap_bytes` is 524,288
(`cost_model.py:362`), 3.1× smaller, and it is tempting to call that an error. It is not.
Setting it to the allocator's real budget makes the model *worse*: softmax RMS 21.1 → 22.2 %,
mean −8.6 → −11.3 %, every other category unchanged. The parameter is the fitted **knee of a
bandwidth derate**, not a capacity, and it earns its 512 KB empirically. The two numbers
measure different things and the shared vocabulary is the problem — rename it
(`lx_derate_knee_bytes`) and document that it is not `_lx_planning_size()`.

## 4. Liveness

`utils.py:84-103` walks `graph.operations` and records, per buffer, the indices at which it
is read or written. Time is **position in the operation list**; an interval is
`[first_use, last_use + 1)` (`plan_solver.py:75-81`); overlap is the obvious comparison
(`plan_solver.py:88-90`). There is no interference graph.

The practical consequence, and one I got wrong before checking: **a dependent chain's
intermediates barely overlap**. In `add6`, `buf0` is dead once `buf1` exists, so six operands
hold about two tiles at a time, not four. Capacity questions must be asked about the *peak
simultaneous* footprint, never the sum.

## 5. The five solvers

| solver | objective | how it picks what to demote |
|---|---|---|
| **`greedy`** (default) | **none** | Chronological bump allocation (`greedy_solver.py:189-198`): step through time, free expiring buffers, allocate starting ones. Whoever arrives when LX is full loses. No size, reuse, or cost awareness. |
| `firstfit` | a priority proxy | Sort by `(span − discount) / uses` ascending (`firstfit_bestfit_solver.py:204-216`) — short-lived, heavily-reused buffers first — then first-fitting gap. |
| `bestfit` | same proxy | Identical ordering; picks the tightest gap instead of the first. |
| `cpsat` | **minimise spilled bytes** | 2-D no-overlap packing with a real objective: `Σ spill_cost·(1 − in_buffer)` where `spill_cost = (read_count + is_intermediate) · size` (`ilp_solver_ortools.py:208-224, 467`). Then a second phase maximises core usage. |
| `simulated_annealing` | fragmentation quality | Anneals over buffer orderings. |

Only CP-SAT expresses "which buffer is worth more", and it expresses it in **bytes**.

## 6. Overflow is silent

There is no re-tiling, no fallback, no warning. Three paths:

1. **Too big on its own** — vetoed before placement, `plan_solver.py:221-229`:
   `min_footprint > limit` → excluded with a reason string.
2. **Fits, but no room at its tick** — the solver simply leaves `address = None`
   (`greedy_solver.py:120-124`; first/best-fit have no `else` branch at all).
3. **Reason recorded, but only at DEBUG.** `allocator.py:194-210` fills `reject_reasons` with
   `"no room on scratchpad (t=…, size=… KB)"`, surfaced solely through `_log_lx_pinning`
   behind `logger.isEnabledFor(DEBUG)`. **No WARNING or ERROR is emitted anywhere when LX
   overflows.**

Nothing upstream reacts: `wsr/` and `work_division.py` contain no LX-capacity logic, so tile
size is never revisited because the working set did not fit.

## 7. What this means for a cost model

Combining with the measurements in `findings_lx.md`:

- **Replacing the default solver is the biggest and cheapest win, and needs no model.** The
  default, `greedy`, is the *worst* of the options measured: time-optimal on only 21/37
  contested bundles, worst regret **+18.2%**. `cpsat` reaches 36/37 at 0.6%. So the win is
  switching away from `greedy` — the default is the defect, not the fix. Note that
  `firstfit`/`bestfit` do NOT help either: their `(lifetime − discount)/uses` ordering
  reproduces greedy's mistake (21/37, +18.2%), because it also reaches softmax's `b1` before
  the more valuable `b2`. Only the CP-SAT objective separates them.
- **A cost model beats the byte proxy only under a specific condition**: some LX-eligible
  candidate carries a *shape-dependent* transport pattern. `cat0`/`stick_scatter` floors at
  44 GB/s and penalised `transpose_outer` at 40, against a 150 default — a 3.75× envelope
  the byte proxy cannot see. Measured regret when it applies: **+36.6%**.
- **The hook is one method.** `spill_cost()` (`ilp_solver_ortools.py:208`) already sits in
  exactly the right place: the CP-SAT objective is `Σ spill_cost·(1 − in_buffer)`. Replacing
  `(read_count + is_intermediate) · size` with a predicted-time delta requires no structural
  change. The alternative, a full `CostModelLayoutSolver(MemoryPlanSolver)`, needs one entry
  in `_PLACEMENT_SOLVERS` (`allocator.py:2108`) and one `Literal` member in `config.py:134`.
- **The model cannot choose alone.** It has no capacity constraint — LX traffic is priced at
  zero, so an unconstrained model would hold everything on chip. The division of labour has
  to be: *the allocator enumerates feasible allocations, the model ranks them.*

## 8. Smaller findings

- `ScratchpadAllocator.__init__` accepts `pre_optimization_passes` / `post_optimization_passes`
  (`allocator.py:116-121`), and `scratchpad/passes.py:21` defines the ABC — but there are no
  implementations and `select_allocator` never passes any. An unused extension seam.
- `_score_layout` (`allocator.py:1448`), the co-optimizer's leaf score, returns raw unpinned
  HBM bytes. `docs/source/compiler/scratchpad_planning.md` already names replacing it with a
  performance model as intended work.
- A `SolveError` falls back to greedy silently (`allocator.py:2209-2219`); a missing
  `ortools` falls back to greedy with a WARNING (`allocator.py:2116-2136`). So a machine
  without OR-Tools gets the worst solver, and only the second case says so.

## 9. Recommended, in order of confidence

1. **Change the default `layout_solver` from `greedy` to `cpsat`.** Justified by measured
   data, no model involved, no new code. `bestfit` is NOT a substitute — measured at 21/37,
   the same as greedy. CP-SAT costs an OR-Tools dependency and a 600 s solve limit, and
   already degrades to greedy with a warning when OR-Tools is absent.
2. **Rename `lx_spill_cap_bytes`.** It is a fitted derate knee, not the LX capacity, and
   the name invites exactly the "fix" that measurably degrades the model.
3. **Emit a WARNING when a buffer is denied LX for want of room.** Today the single most
   consequential allocation event in the compiler is invisible above DEBUG.
4. **Only then** consider a cost-model objective, gated on the trigger condition in §7.
