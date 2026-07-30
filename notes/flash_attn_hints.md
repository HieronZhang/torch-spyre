# Flash-attention on Spyre: what each `spyre_hint` does, and why it matters for the cost model

*Source: `flash_attn_example.py` (the `flash_spyre` path), IR dump
`haoyang_logs/flash_attn_ir.log`. This explains each hint so a sweep over their values
gives us relative-performance data for a real multi-op coarse-tiled program.*

## What the program computes

`flash_spyre` is one **fused** online (flash) attention pass over
`Q,K,V ∈ [B,H,L,D]` with an additive causal mask, `B=1, H=32, Lq=Lk=4096, D=128`.
Unlike `flash_cpu` (which loops in Python over blocks), `flash_spyre` writes the whole
computation *once* and lets the compiler tile it — the Python is block-loop-free; the
five `spyre_hint` context managers are what turn it into a tiled loop nest. The IR
lowers it to **28 ops** fused under one coarse-tiled loop, the important ones being:

- `scores = (Q·scale) @ (K·scale)ᵀ` — a batched matmul, **reduction over `D=128`**
  (IR: `reduction_type=batchmatmul, reduction_ranges=[128]`), output `[B,H,Lq,Lk]`.
- `scores += mask`; `block_max = amax(scores, dim=Lk)` (`reduction_ranges=[4096]`);
  `running_max = max(real_max, block_max)`; `exp_scores = exp(scores − running_max)`;
  `correction = exp(real_max − running_max)`.
- `denominator = denominator·correction + sum(exp_scores, dim=Lk)` (`reduction=sum,
  ranges=[4096]`).
- `output = output·correction + exp_scores @ V` — the second batched matmul,
  **reduction over `Lk=4096`** (`reduction_type=batchmatmul, reduction_ranges=[4096]`),
  output `[B,H,Lq,D]`.
- final `output / denominator`.

So it is **two matmuls with different reduction dims (D and Lk) glued together by the
online-softmax reductions** — a genuinely multi-op coarse-tiled program, and exactly the
kind of case (beyond the softmax family) we want in the database.

## The five hints

The hints nest (outer → inner): `tiles B` ⊃ `tiles H` ⊃ `tiles Lq` ⊃ `tiles Lk` ⊃
`work_div`. There are **two distinct kinds**:

### 1–4. `spyre_hint(tiles={dim: N})` — COARSE TILING (the sequential loop nest)

Each `tiles={dim: N}` chops `dim` into **N sequential blocks** processed one at a time,
carrying the running softmax state (`real_max`, `denominator`, `output`) across blocks.
`N` is the **tile count**; the block size is `dim / N`. In the example these come from
the `*_block_size` variables:

| hint | value | tile count N | block size | effect |
|---|---|---:|---:|---|
| `tiles={"B": B // b_block_size}` | `1 // 1` | **1** | 1 | B not tiled (B=1) |
| `tiles={"H": H // h_block_size}` | `32 // 4` | **8** | 4 heads | H tiled into 8 |
| `tiles={"Lq": Lq // q_block_size}` | `4096 // 1024` | **4** | 1024 queries | Lq tiled into 4 |
| `tiles={"Lk": Lk // kv_block_size}` | `4096 // 4096` | **1** | 4096 | Lk **not** tiled (see FIXME) |

The IR confirms this exactly: `CoarseTileInfo(loop_count=[8, 4],
loop_tiled_dims=[[1],[2]])` — the outer loop tiles **dim 1 = H into 8**, the inner loop
tiles **dim 2 = Lq into 4**, giving **32 sequential tile iterations**; B and Lk are not in
the loop. So the fused kernel streams 32 tiles, each an `[H=4, Lq=1024, Lk=4096, D=128]`
slice of the attention, reusing the on-chip (LX) intermediates within a tile.

**What the tile counts trade off** (the knob the cost model must eventually rank):
- **Fewer, bigger tiles** (small N): fewer loop iterations and less recomputation of the
  running max/denominator, but a **larger per-core working set** — the `[Lq_tile, Lk]`
  scores block is the big intermediate, and if it overflows LX it spills to HBM and runs
  at the derated rate (the §15 LX-spill effect). `q_block_size = Lq//2` (N=2) compiles
  faster but each tile's scores block is 2× bigger.
- **More, smaller tiles** (large N): each tile fits LX comfortably, but there are more
  loop iterations (fill/drain overhead per tile, §16 underfill) and the online-softmax
  correction is applied more times.
- **`Lk` is special:** tiling `Lk` (the key/value length) is what makes this "flash" —
  it bounds the scores block to `[Lq_tile, Lk_tile]`. The example currently forces
  `kv_block_size = Lk//1` (N=1, no Lk tiling) with the note *"FIXME: current limitation
  disallows coarse tiling in Lk"*, so today the full `Lk=4096` scores row is materialized.
  Sweeping `FA_LK_TILES>1` is a useful probe of whether/when that limitation triggers.

### 5. `spyre_hint(work_div={dim: N})` — INTRA-TILE CORE PARALLELISM

`work_div={"H": 4, "Lq": 8, "Lk": 8}` is **not** tiling; it asks the planner how to split
**one tile's** work across the (≤32) cores — the same mechanism as a matmul `WD_M/N/K`
split, but named by logical dim. The IR shows the **realized** split in
`op_it_space_splits`: the main fused ops run `{d0: 4, d1: 8}` = **32 cores** (H-within-tile
×Lq-within-tile = 4×8), with the reduction/`Lk` axis kept whole (`d2: 1`) — the requested
`Lk: 8` is *not* separately realized because 4×8 already saturates 32 cores. The
sparse-init reductions that build `real_max`/`denominator` run `{d0:1, d1:32, d2:1}`
(a 32-way split over H before the loop).

So `work_div` sets **how parallel each tile is** (and therefore per-core tile height →
the §16 underfill and §9 systolic fill), while `tiles` sets **how many tiles run in
sequence**. Together they determine the per-core working set = the LX-vs-HBM question.

## Sweepable knobs (for the sweep, `_flash_attn_workload` in `profile_ops.py`)

| env | meaning | default | legal values |
|---|---|---:|---|
| `FA_B/FA_H/FA_LQ/FA_LK/FA_D` | shape | 1/32/4096/4096/128 | H,Lq,Lk powers of 2 |
| `FA_B_TILES` | B tile count | 1 | 1 (B=1) |
| `FA_H_TILES` | H tile count (block = H/N) | 8 | divisors of H: 1,2,4,8,16,32 |
| `FA_LQ_TILES` | Lq tile count (block = Lq/N) | 4 | divisors of Lq: 1,2,4,8,16 |
| `FA_LK_TILES` | Lk tile count | 1 | 1 today; >1 probes the FIXME |
| `FA_WD` | work_div, e.g. `"H:4,Lq:8,Lk:8"` | `H:4,Lq:8,Lk:8` | product realized ≤ 32 cores |

## Why this matters for the cost model (the end goal)

The final goal is to give **relative** performance guidance as these block sizes change,
so the compiler (or an engineer) can pick the best tiling for a real LLM deployment. To
do that the model must read the **loop-level IR** and estimate:

- the **loop trip count** (`CoarseTileInfo.loop_count` product — here 8×4=32),
- the **per-core working set** per tile (the scores `[Lq_tile, Lk_tile]` block dominates)
  → LX-resident vs HBM-spilled (§15),
- the **per-core tile height** from `op_it_space_splits` → underfill (§16) and matmul
  systolic fill (§9),
- the **two matmuls' MACs and reduction dims** (D and Lk) and how they overlap with the
  memory traffic (§10 — the very overlap model we are re-examining),
- the online-softmax reductions' HBM/LX traffic and any **read-after-write** across the
  loop-carried state (`real_max`/`denominator`/`output`) when it spills (§3).

We are **not** modeling flash attention now — the single-op, matmul, and coarse-tiling
terms must be solid first, and the model is not yet usable here (it under-counts multi-op
coarse programs, cf. `softmax_unrolled`). What we *can* do now is **sweep the hint values
and record measured time + IR + cost-model features**, which (a) builds the database with
many multi-op coarse-tiling cases beyond the softmax family, and (b) gives us the ground
truth to validate a future flash-attention cost term against. That sweep is the `FLASH`
section of `run_outlier_sweep.sh`.
