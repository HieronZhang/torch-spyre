# Research: cost-model-driven compilation for Spyre

Work done while the accelerator was down. **Everything here is predicted, not measured** —
except where it cites the recovered measurement database (commit `3f57eb5`, 2828 records),
which is stated explicitly at each point.

## Read in this order

| file | what it answers |
|---|---|
| `flash_lx_findings.md` | Flash attention: CP-SAT is optimal, the DEFAULT solver is 20-28% off. |
| `case_studies.md` | The four Granite programs, and where their tiling x LX design space is large. |
| `findings_lx.md` | What is the LX allocation choice worth, and can a cost model improve it? The evidence. |
| `lx_allocator.md` | How does LX allocation work today? Source-verified deep dive. |
| `lx_policy.md` | Should we build it? Design note and recommended sequence. |
| `literature_review.md` | Where the field is: 16 sections, ~225 sources, 479 links, papers and lab blogs. |
| `literature_review_critique.md` | What the completeness critic found wrong and missing in the first draft. |

### Research directions the review lands on (its §12)

| | question | status |
|---|---|---|
| Q1 | Does analytical *ranking* survive an opaque backend on a NoC dataflow chip? | narrowed — TpuGraphs already did this with a *learned* model |
| Q2 | Does LX spilling bind on **capacity or layout legality**? | **open, cheapest, best-posed** — tt-mlir found spills are constraint-driven with 40–94 % headroom left |
| Q3 | Joint tile × LX under a *measured* objective, work division fixed | restated — Welder/Stream/DeFiNES already co-optimise; the per-core partition is the novelty |
| Q4 | Is the FlashAttention dataflow even the right target on Spyre? | open — FlatAttention reports 4.1× from *replacing* it with NoC collectives on tile hardware |
| Q5 | Can predicted-latency `score_fusion` beat Inductor's byte proxy? | reframed — XLA PriorityFusion already does this; the question is Inductor integration |
| Q6 | Where is the knee of the 2 MiB/core LX capacity curve? | open and cheap |
| Q7 | How should bucket/pad sets be chosen under a serving SLO? | **new** — vllm-spyre defines buckets by hand; no cited system chooses them with a model |
| Q8 | Does block-scaled FP8 survive 128-byte stick alignment? | **new** — MX fixes block 32; literature coverage of the interaction is zero |
| Q9 | What is the rank-quality / compile-time Pareto front? | **new** — everyone reports one or the other, never the front |

Q2 and Q6 are the ones a team with one accelerator could start on tomorrow.

**The most Spyre-relevant thing in the review:** IBM already shipped the ancestor of half
this problem. onnx-mlir's NNPA backend lowers ONNX through ZHigh/ZLow with explicit
`stickify`/`unstickify` over zDNN's stick layout — the origin of the term this compiler uses
— and it already carries a measurement-fitted analytical cost model for stick hardware
(`PerfModelArch14/15.inc`, with r² in the source). Its *worst* fits are Unstick (0.429),
Stick (0.691) and MatMul_3ds (0.706): the layout conversions are the hardest terms to model,
which is both a caution and a reassurance for this project. Separately, DNNDaSher reports
1.27–4.12× (avg 2.3×) end-to-end on AIU purely from eliminating and coarsening layout
shuffles.

## Tools (all run with no hardware)

```sh
python3 research/lx_choice.py --validate       # mutation harness vs 56 measured on/off pairs
python3 research/lx_choice.py --contested      # bundles where the LX choice matters
python3 research/lx_choice.py --policies       # cost model vs the compiler's own heuristics
python3 research/lx_choice.py --monotonicity   # does more LX residency ever predict SLOWER?
python3 research/bytes_vs_time.py              # when can time-ranking beat byte-ranking?
python3 research/workloads.py --check          # Granite case studies, on CPU
python3 research/design_space.py               # tiling x LX design space per program
python3 research/lx_experiment.py --exhaustive # proven-optimal vs the shipped solvers
```

`compile_only.py` is the bridge for the run machine — it compiles a workload with **no
device** (`IS_INDUCTOR_SPAWNED_SUBPROCESS=1` + `FakeTensorMode`) and dumps features as JSON
for `lx_choice.py --records`. **It has never been run**: this machine has no `torch_spyre._C`
and no SDK. Its first execution is an experiment, and the one-line check it prints first
decides whether the whole offline-compile path works.

## Headline findings

1. **The default LX solver leaves 18-28% on the table.** `greedy` is time-optimal on 21/37
   softmax bundles and `firstfit`/`bestfit` are no better; on flash attention, where
   exhaustive search proves the optimum, `greedy` is 20.4% and 28.3% off. `cpsat` is exactly
   optimal on both corpora. A one-line config change, no cost model involved. Highest-value
   result here, and now confirmed on two independent corpora.
2. **For LX residency the existing byte proxy is usually *exact*, not approximate.** When all
   candidates move at the same bandwidth, minimising spilled bytes is minimising time.
3. **The exception is precise and worth 36.6%** — a candidate carrying `stick_scatter` or
   penalised `transpose_outer`, which run 3.4–3.75× below default bandwidth. Cheap to test
   for, and reachable in practice (a materialised transpose lowers to `clone`, which is
   LX-eligible).
4. **The model is not monotone in LX residency** on 14/37 bundles — more residency can
   predict *slower*, because spilling a broadcast operand flips the bundle onto a cost
   formula that omits the turnaround term. A prerequisite for any ranker, and a model bug
   independent of LX.
5. **LX overflow is completely silent** — no warning above DEBUG on the compiler's most
   consequential allocation event.
6. **Tile size is the larger opening.** No performance-driven tile-size search exists
   anywhere in the compiler, so there is no incumbent heuristic to beat — unlike LX.

## Two corrections made during this work

Recorded because the reasoning is more useful than the conclusions.

- I derived a bound saying a cost model could essentially never beat the byte proxy, from a
  bandwidth envelope of 1.33×. That envelope came from two flat constants and omitted the
  shape-dependent transports, which reach 40 GB/s. The true envelope is 3.75× and the
  conclusion reverses.
- I flagged `lx_spill_cap_bytes` (512 KB) as a 3.1× error against the allocator's real
  1587 KB budget. Setting it to the true value makes softmax *worse*. It is a fitted derate
  knee, not a capacity; the fix is to rename it.

Both were caught by testing a claim rather than reasoning from it.

## Not done

- Nothing has been compiled. The Granite workloads are CPU-verified for shape and numerics
  only; whether their coarse-tiling hints are accepted is unknown.
- No measurement. Every performance statement is a model prediction.
- The trigger condition in finding 3 is shown by perturbing recorded features, not by a real
  program hitting it.
