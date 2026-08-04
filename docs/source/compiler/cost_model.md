# Analytical Cost Model

The cost model predicts how long a compiled graph will take to run, from the loop-level
IR, without running it. It exists so the compiler can compare two candidate plans —
different coarse tile sizes, different work divisions, different scratchpad placements —
cheaply enough to do it during compilation.

It is **off by default** and adds nothing to a normal compile.

> **Not to be confused with** `work_division.cost_model_matmul_division`, an unrelated
> model that chooses a matmul work division. This page describes the whole-program
> runtime model in `cost_model.py` and the reporting pass in `cost_model_pass.py`.

## Where it runs

The pass runs immediately after the pre-scheduling pipeline, at the point where the
loop-level IR is final: layouts resolved, restickify inserted, coarse tiling applied, work
division committed, scratchpad placement done. Nothing after this changes what the program
moves or computes — only how it is packaged into kernels.

```text
                        torch.compile
                              │
                              ▼
   FX graph ──▶ Inductor lowering ──▶ graph.operations   (loop-level IR)
                                              │
     ┌────────────────────────────────────────▼────────────────────────────┐
     │                    CustomPreSchedulingPasses                        │
     │  deadcode ▸ split_multi_ops ▸ layouts ▸ restickify ▸ coarse_tile    │
     │  ▸ work_division ▸ scratchpad_planning                              │
     └────────────────────────────────────────┬────────────────────────────┘
                                              │
                          ═══════════ IR IS FINAL HERE ═══════════
                          bytes, tiling and core division all settled
                                              │
                            dump_loop_ir  ◀───┤   SPYRE_DUMP_IR
                                              │
                          cost_model_pass ◀───┤   SPYRE_DUMP_COST ── the switch
                                  │           │   unset/"0" ▸ returns immediately,
                                  │           │      no extraction, no cost
                                  │           │   "1"       ▸ report
                                  │           │   "2"       ▸ report + check below
                                  │           │
      ┌───────────────────────────┴──────┐    │
      │ group ops into the kernels the   │    │
      │ backend will fuse, then price    │    │
      │ each ONCE with cost_model.py     │    │
      └───────────────────────────┬──────┘    │
                                  ▼           │
                        ┌──────────────────┐  │
                        │   CostReport     │  │      ┌─── printed breakdown
                        │  .total_us   ────┼──┼─────▶│
                        │  .groups[]       │  │      └─── returned to the caller:
                        └──────────────────┘  │           stored as last_cost_report,
                                  │           │           so another pass can compare
                                  │           │           plan A vs plan B without
                                  │           │           compiling or running either
                                  ▼           ▼
                              Inductor Scheduler
                                              │
                                              ▼
                                      spyre_fuse_nodes        real kernels exist now
                                              │
                verify_against_fused_nodes ◀──┤  only at "2": re-price using the REAL
                                              │  bundles and print the gap against the
                                              ▼  estimate above
                                          SuperDSC ──▶ DeepTools ──▶ device
```

The switch is the whole design: at `""` or `"0"` the pass returns before touching the
graph, so a normal compile pays one attribute read. Everything else — extraction, pricing,
printing — happens only when it is asked for.

## Turning it on

Set `SPYRE_DUMP_COST`, or patch the config directly:

| value | effect |
|---|---|
| unset, `""`, `0`, `false`, `off` | disabled — the pass returns before touching the graph |
| `1`, `true`, `yes`, `on` | print a per-kernel breakdown and the program total after pre-scheduling |
| `2` | also re-score against the real fusion bundles and report the difference |

> `SPYRE_DUMP_COST` is also read directly by the older per-operation dump in
> `dump_cost_model.py`, which recognises only `1`/`true`/`yes`/`on`. At `2` this pass runs
> but that dump does not — and `profile_ops.py` reads its output, so use `1` when running
> the benchmark sweeps.

```bash
SPYRE_DUMP_COST=1 python my_model.py
```

```python
from torch_spyre._inductor import config

with config.patch({"cost_model": "1"}):
    compiled = torch.compile(model)
```

## What it prints

```text
predicted total:      336.8 us over 1 kernel(s), 5 op(s)

      kernel       loops   trip   predicted us  ops
           0           0      8          336.8  5
                                         112.3    amax (16.8 MB)
                                         112.3    sub (16.8 MB)
                                         112.3    div (16.8 MB)
                                           0.0    exp (on-chip only, 4.2 MB)
                                           0.0    sum_1 (on-chip only, 2.2 MB)
```

## Using the result

The pass returns a `CostReport` rather than only printing, so another pass or an external
tool can compare plans:

```python
from torch_spyre._inductor.cost_model_pass import cost_model_pass

report = cost_model_pass(graph)          # None when disabled
if report is not None and report.total_us < best_so_far:
    ...
```

It is also stored on the pipeline instance as `last_cost_report`, and in the module global
`cost_model_pass.LAST_REPORT`.

| field | meaning |
|---|---|
| `CostReport.total_us` | predicted runtime for the whole graph — the number to compare |
| `CostReport.groups` | one entry per kernel, in program order (only the printout sorts by cost) |
| `CostReport.from_fused_nodes` | `True` when built from real post-fusion bundles |
| `GroupCost.predicted_us` | that kernel's predicted time |
| `GroupCost.loop_group_ids` | the coarse-tiling loops inside this kernel, for labelling |
| `GroupCost.ops` | per-operation attribution — see the second limitation below |

## How a kernel is priced

`predict_ops` is defined over **one fused kernel** and is deliberately *not* additive over
its operations: it de-duplicates external inputs shared inside a bundle, and its turnaround
and overlap terms are `min`/`max` reductions over bundle totals. Across the 521 recorded
multi-operation programs, pricing operations separately and summing differs from pricing
the bundle by **−94 % to +33 %** — usually an *under*-count. Getting the grouping right is
therefore not a presentation detail; it decides whether `total_us` means anything.

So the pass groups the way the backend fuses. `spyre_fuse_nodes` accumulates every
*contiguous run* of Spyre nodes into one bundle with no size limit, and the pass mirrors
that in three respects:

- a run is broken by any operation **not on the Spyre device** — a CPU operation's buffer is
  still a `ComputedBuffer`, so testing the type alone would both fail to break the kernel and
  price CPU work as Spyre traffic;
- a run is broken by any operation the extractor cannot model;
- when `bundle_symbolic_args` is off the backend does **not fuse at all**, and neither does
  the report — every operation becomes its own kernel.

Coarse-tiling `loop_group_id` is read only to *label* the loop structure inside a kernel,
not to decide boundaries.

## Two limitations worth knowing

**Kernel boundaries are an estimate.** Real bundles are formed later, by `spyre_fuse_nodes`,
which sees scheduler nodes rather than IR operations. Contiguity is a close proxy, not a
guarantee. Setting the flag to `2` measures the gap directly rather than assuming it is
small: it re-scores after fusion, using the real bundles, and prints both totals and the
difference.

**Per-operation times are an attribution, not predictions.** Each kernel is priced once, as
a whole; the per-op column then splits that total by each operation's share of the
main-memory bytes. The parts sum to the kernel total by construction, but they are not
separately meaningful. Two known distortions: the split carries no compute term, so it
misattributes a compute-bound kernel; and the weights are per-operation while the total
de-duplicates shared inputs, so a share can be off by up to a third of itself (measured on
recorded softmax bundles: 33.3 % where a consistent split gives 25.0 %). An operation whose
output stays on chip shows `0.0` because it adds no traffic of its own — it was fused away,
not free.

## Accuracy

The model is documented and scored in `notes/cost_model_report.md`, which derives every
term and states how well each is understood. Current accuracy by category — pointwise
8.3 %, broadcast 5.7 %, reduction 7.2 %, transport 6.1 %, matmul 15.1 %, coarse tiling
7.4 % RMS.

For plan *ranking* rather than absolute accuracy, see `notes/cost_model_directions.md`: over
26 measured coarse-tiling ladders the model picks a tile count tied for best on 17, with a
median regret of 0 %. That document also records where it is blind (tiled softmax, where
the prediction stops depending on tile count) and where it is wrong (row-tiled matmul,
where it turns up one rung too early).

Flash attention is **not** currently modelled correctly — it is mispredicted by more than
an order of magnitude from a rank-4 extraction bug, diagnosed in the same document.

## Implementation

| file | role |
|---|---|
| `cost_model.py` | the model itself: pure Python, no torch dependency, importable standalone |
| `dump_cost_model.py` | IR → `OpFeatures` extraction, and the older per-op feature dump |
| `cost_model_pass.py` | the pass: grouping, per-group pricing, the report |
| `tests/inductor/test_cost_model_pass.py` | grouping, attribution and disabled-path guards |

The pass never raises. Instrumentation that can break a compilation is worse than no
instrumentation, so every entry point catches broadly and logs instead.
