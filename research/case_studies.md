# Granite case studies: where the tiling × LX design space is actually large

Four real-model programs at Granite dimensions, and the design space each one presents to
the compiler. **Two of them are genuinely contested** — capacity binds, several allocations
are feasible, and they differ by 2.0–2.6× in spilled traffic. One is not contested at any
tile count, which is itself worth knowing.

The most useful result is the *interaction*: on the MLP, changing the tile count from 2 to 4
eliminates the spill entirely, so it beats every possible residency decision at 2 tiles. The
tile-size choice dominates the allocation choice at that boundary.

---

## 1. What was built, and its status

`research/workloads.py` — four programs, dimensions from `tests/resource/models/granite-*.yaml`
(`d_model` 4096, `d_ff` 12800, `head_dim` 128):

| program | what it is | status |
|---|---|---|
| `swiglu_mlp` | `down(silu(x @ Wg) * (x @ Wu))` | runs on CPU, shapes and fp16 numerics verified |
| `rmsnorm_residual` | `x · rsqrt(mean(x²)+ε) · w + r` | same |
| `attention_block` | `bmm(softmax(q·kᵀ), v)` with an explicit transpose | same |
| `transformer_block` | norm → MLP → residual | same |

**None has been compiled.** They have never been through the Spyre pipeline, so whether
their coarse-tiling hints are accepted, and what the extractor actually emits for them, is
unknown. Everything below is computed analytically from shapes and the dependence order, not
from compiled features.

## 2. The design space

`research/design_space.py` computes, per tile count: the per-core footprint of each
intermediate, the peak simultaneous demand under the allocator's own liveness model, which
subsets fit in the 1587 KB/core budget, and the HBM traffic each choice adds.

```sh
python3 research/design_space.py
```

### SwiGLU MLP — `seq=2048`, `d_ff=12800`

| tiles | largest tile | peak if all resident | all fit? | feasible choices | best spill | worst spill | spread |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 1600 K | 4800 K | no | 1 | 400 M | 400 M | — |
| **2** | **800 K** | **2400 K** | **no** | **6** | **200 M** | **400 M** | **2.00×** |
| 4 | 400 K | 1200 K | yes | 16 | 0 M | 400 M | — |
| 8 | 200 K | 600 K | yes | 16 | 0 M | 400 M | — |

**Contested at 2 tiles.** Three full-width `d_ff` intermediates are live across the
elementwise stage; two of them overlap at the peak. At 2 tiles the peak is 2400 K against a
1587 K budget, so something must spill, six subsets are feasible, and the choice is worth 2×.

### Transformer block — `seq=1024`

| tiles | largest tile | peak if all resident | all fit? | feasible choices | best spill | worst spill | spread |
|---:|---:|---:|---|---:|---:|---:|---:|
| **1** | **800 K** | **2400 K** | **no** | **96** | **100 M** | **256 M** | **2.56×** |
| 2 | 400 K | 1200 K | yes | 256 | 0 M | 256 M | — |

**Contested untiled.** Eight intermediates of three different kinds — a cross-row reduction
result, full-width activations, and matmul operands — compete for the same scratchpad. 96
feasible allocations, 2.56× between best and worst. This is the case with heterogeneous
access patterns, and therefore the one where the byte proxy and a cost model could actually
diverge (`findings_lx.md` §6).

### RMSNorm + residual — not contested

Peak demand is 1024 K even untiled, under the 1587 K budget, so everything fits at every tile
count and the allocator has no decision to make. A norm's intermediates are simply too small
to compete at `d_model` = 4096. Worth recording as a negative result: not every real program
presents a design space.

## 3. The interaction: when a smaller tile beats a better allocation

This is the part that cannot be seen by varying residency at fixed tiling.

On the MLP, the *best possible* allocation at 2 tiles still spills **200 MB**. At 4 tiles
nothing spills at all — **0 MB** — because halving the tile halves each intermediate's
per-core footprint and the whole working set drops under the budget.

So at that boundary the tile-size decision dominates the residency decision: no allocation
choice at 2 tiles can reach what the default choice at 4 tiles gets for free. That is a
concrete instance of *"maybe a smaller tile size is better"*.

**It is not free, and that is the interesting part.** More tiles means more loop iterations
and a shorter per-core tile, which the model derates through `coarse_underfill_eff`. Here the
rows per core are 32 at 2 tiles and 16 at 4, both near the point where that derate saturates,
so the trade looks favourable — but "looks favourable" is doing real work in that sentence.
Settling it needs the cost model on compiled features, because it is exactly a trade of
spill traffic against underfill, and those are different terms of the model.

**This is the research question the study should have been aimed at**: not "which tensor goes
in LX", which CP-SAT already answers well, but "which tile count, given that tile count
decides whether the LX question arises at all". Nothing in the compiler chooses tile counts
today — they come from user hints — so unlike LX residency there is no incumbent heuristic to
beat.

## 4. What these results are not

- **Not measured.** No hardware, no compilation.
- **Not predicted times.** The traffic columns are the byte proxy — the same quantity CP-SAT
  optimises — because predicted time needs the extractor's `OpFeatures` and these programs
  have never been extracted. Where traffic and time diverge is `findings_lx.md` §6.
- **Not validated against the real allocator.** The liveness and footprint model here mirrors
  `scratchpad/utils.py:84` and `plan_solver.py:75-81`, and it reproduced the recorded
  softmax bundles correctly, but it has not been checked against a compiled Granite program.
- **The attention block is absent from the table** because its intermediates are
  shape-dependent on the softmax's internal decomposition, which analytic shapes do not pin
  down. It needs compilation.

## 5. To finish this properly

With the card back, one command per program closes all four gaps:

```sh
IS_INDUCTOR_SPAWNED_SUBPROCESS=1 python3 research/compile_only.py \
    --workload swiglu_mlp --tiles 2 --out mlp_t2.json
python3 research/lx_choice.py --records mlp_t2.json --contested --policies
```

That would give real extracted features, real predicted times, and would test the one claim
in `findings_lx.md` §6 that has never been checked on a real program: whether a shipping
workload puts a slow-transport buffer into LX contention.
