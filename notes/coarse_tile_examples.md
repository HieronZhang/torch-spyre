# Coarse-tiling examples (`coarse_tile/`)

Each `run_*.py` is a standalone script that runs one op on the Spyre device (compiled
path) and compares against CPU. Each declares named dims (`declare_tensor_dim` /
`name_tensor_dims`) and requests **coarse tiling** with
`spyre_hint(num_tiles_per_dim={"<DIM>": <count>})` — the pass splits that dim into
`<count>` tiles so the intermediate(s) stay in on-chip LX. `utils.py` holds the shared
`compare_with_cpu` harness; `run_all.py` runs all of them and reports pass/fail.

The tiling **dimension** is what distinguishes the variants:

- **reduction-dim tiling** — tile the axis being reduced/contracted (batch `B` for a
  reduction, `K` for a matmul). Splits the work; a per-tile partial result is combined.
- **output-dim tiling** — tile an output axis (`M` rows, or the softmax `NROW`). Each
  tile is an independent slice of the output.
- **nested** — two hints, an outer output-dim tile and an inner reduction-dim tile.

## Reductions over dim 0

| script | op | shape | tiled dim |
|---|---|---|---|
| `run_sum_dim0_tiled.py` | `x.sum(dim=0)` | `[B=512, D=64] → [D]` | `B` ×4 (the reduced dim) |
| `run_amax_dim0_tiled.py` | `x.amax(dim=0)` | `[512, 64] → [64]` | `B` ×4 |
| `run_amin_dim0_tiled.py` | `x.amin(dim=0)` | `[512, 64] → [64]` | `B` ×4 |

Reduce the 512-row batch to a length-64 vector, tiling the reduced `B` axis into 4
chunks; each chunk's partial reduction is combined. `D=64` stays on one stick.

## Matmul / mm (2-D)

| script | op | shape | tiled dim | what it isolates |
|---|---|---|---|---|
| `run_matmul_k_tiled.py` | `a @ b` | `[64,512] @ [512,32]` | `K` ×4 | contraction-dim tiling |
| `run_mm_k_tiled.py` | `torch.mm` | `[64,512] @ [512,32]` | `K` ×4 | same, explicit `mm` |
| `run_matmul_row_tiling.py` | `x @ y` | `[256,128] @ [128,64]` | `M` ×4 | **output-row** tiling |
| `run_mm_nested_outer_M_inner_K.py` | `torch.mm` | `[128,512] @ [512,32]` | `M` ×2 **then** `K` ×4 | nested output+reduction |

`*_k_tiled` splits the reduction `K` (128 elems/tile = 2 sticks at fp16) — each tile is a
partial product, summed. `matmul_row_tiling` instead splits output rows `M` (independent
row slices; this is the `matmul_row_tiling` op the cost-model report §16 discussed).
`nested_outer_M_inner_K` does both, one hint inside the other.

## Batched matmul (bmm)

| script | op | shape | tiled dim | note |
|---|---|---|---|---|
| `run_bmm_k_tiled.py` | `torch.bmm` | `[B=8,64,512] @ [8,512,32]` | `K` ×4 | full 3-D × 3-D batch |
| `run_bmm_3d2d_k_tiled.py` | `torch.matmul` | `[8,64,512] @ [512,32]` | `K` ×4 | **2-D weight shared** across batch |
| `run_bmm_nested_outer_B_inner_K.py` | `torch.bmm` | `[B=4,64,512] @ [4,512,32]` | `B` ×2 **then** `K` ×4 | tile the batch, then `K` |

All three are `B` independent `[M,K]@[K,N]` products (`B·M·N·K` MACs total). The `3d2d`
variant broadcasts one 2-D weight `[K,N]` across the batch (weight loaded once, not
`B×`). `nested_outer_B_inner_K` tiles the batch dimension itself (2 tiles) with an inner
`K` split.

## Softmax

| script | op | shape | tiled dim | mode |
|---|---|---|---|---|
| `run_softmax_row_tiling.py` | `torch.softmax(dim=-1)` | `[16384, 4096]` | `NROW` ×4 | fused softmax, multi-core, LX planning |
| `run_softmax_unrolled.py` | manual `amax→sub→exp→sum→div` | `[256, 64]` | `B` ×4 | `unroll_loops=True`, `sencores=1` |

Both compute softmax over the last dim with the 5 ops fused so the intermediates live in
LX. The difference is **how the tile loop is realised in the IR**:

- **row_tiling**: emits a coarse-tile *loop* —
  `loop_info=CoarseTileInfo(loop_count=[4], loop_tiled_dims=[[0]])`. This is the shape the
  cost model's coarse-tiling features key on (report §13–§15).
- **unrolled** (`config.unroll_loops=True`): the tile loop is **unrolled away** — there is
  **no `loop_info`/`CoarseTileInfo`**; the split shows up only as
  `dim_hints=[DimHint(split_count=4, loop_var=None)]` on a `FixedTiledLayout`. Small,
  single-core (`sencores=1`).

> **Cost-model note.** Both IR forms now also carry a per-buffer `allocation={'lx': <bytes>}`
> — the actual LX allocation (the ground-truth the report §14 wanted instead of the
> working-set proxy). The **unrolled** form has no `CoarseTileInfo`, so the current feature
> extractor (which derives `loop_trip`/`tiles_output_dim` from `loop_count`) does not yet
> see it as coarse-tiled — it would need to read `dim_hints` instead.
