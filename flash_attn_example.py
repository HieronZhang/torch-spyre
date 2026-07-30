import math

import torch
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.propagate_named_dims import (
    declare_tensor_dim,
    name_tensor_dims,
)

SEED = 42

B, H, D = 1, 32, 128
Lq = 4096
Lk = 4096

q_block_size = Lq // 4  # replace by 'Lq // 2' for faster compilation time

# FIXME: current limitation disallows coarse tiling in Lk
kv_block_size = Lk // 1

h_block_size = 4  # replace by 'H // 2' for faster compilation time
b_block_size = 1


def flash_cpu(queries, keys, values, mask):
    scale = 1.0 / math.sqrt(math.sqrt(D))

    output = torch.zeros_like(queries)
    real_max = torch.full(
        (B, H, Lq), float("-inf"), device=queries.device, dtype=torch.float16
    )
    denominator = torch.zeros((B, H, Lq), device=queries.device, dtype=torch.float16)

    for b_start in range(0, B, b_block_size):
        b_end = b_start + b_block_size
        for h_start in range(0, H, h_block_size):
            h_end = h_start + h_block_size

            for lq_start in range(0, Lq, q_block_size):
                lq_end = lq_start + q_block_size
                queries_tile = queries[b_start:b_end, h_start:h_end, lq_start:lq_end]
                real_max_tile = real_max[b_start:b_end, h_start:h_end, lq_start:lq_end]
                denominator_tile = denominator[
                    b_start:b_end, h_start:h_end, lq_start:lq_end
                ]
                output_tile = output[b_start:b_end, h_start:h_end, lq_start:lq_end]

                for lk_start in range(0, Lk, kv_block_size):
                    lk_end = lk_start + kv_block_size
                    mask_tile = mask[:, :, lq_start:lq_end, lk_start:lk_end]
                    keys_tile = keys[b_start:b_end, h_start:h_end, lk_start:lk_end]
                    values_tile = values[b_start:b_end, h_start:h_end, lk_start:lk_end]
                    keys_tile_T = keys_tile.transpose(-1, -2).contiguous()

                    scores = torch.matmul(
                        queries_tile * scale, keys_tile_T * scale
                    )  # tile_b, tile_h, tile_lq, tile_lk
                    scores = (
                        scores + mask_tile
                    )  # additive mask in [.., tile_lq, tile_lk]
                    block_max = torch.amax(scores, dim=-1)  # tile_b, tile_h, tile_lq
                    running_max = torch.maximum(
                        real_max_tile, block_max
                    )  # tile_b, tile_h, tile_lq

                    exp_scores = torch.exp(
                        scores - running_max.unsqueeze(-1)
                    )  # tile_b, tile_h, tile_lq, tile_lk
                    correction = torch.exp(
                        real_max_tile - running_max
                    )  # tile_b, tile_h, tile_lq

                    denominator_tile.copy_(
                        denominator_tile * correction + exp_scores.sum(dim=-1)
                    )  # tile_b, tile_h, tile_lq
                    output_tile.copy_(
                        output_tile * correction.unsqueeze(-1)
                        + torch.matmul(exp_scores, values_tile)
                    )  # tile_b, tile_h, tile_lq, D

                    real_max_tile.copy_(running_max)

    return output / denominator.unsqueeze(-1)


def flash_spyre(queries, keys, values, mask):
    scale = 1.0 / math.sqrt(math.sqrt(D))

    output = torch.zeros_like(queries)

    # FIXME: create a sparse real_max tensor via reduction
    real_max = torch.full(
        (B, H, Lq, 64), float("-inf"), device=queries.device, dtype=torch.float16
    )
    real_max = real_max.amax(dim=-1)  # B, H, Lq sparse

    # FIXME: create a sparse denominator tensor via reduction
    denominator = torch.zeros(
        (B, H, Lq, 64), device=queries.device, dtype=torch.float16
    )
    denominator = denominator.amax(dim=-1)  # B, H, Lq sparse

    with spyre_hint(tiles={"B": B // b_block_size}):
        with spyre_hint(tiles={"H": H // h_block_size}):
            with spyre_hint(tiles={"Lq": Lq // q_block_size}):
                with spyre_hint(tiles={"Lk": Lk // kv_block_size}):
                    with spyre_hint(work_div={"H": 4, "Lq": 8, "Lk": 8}):
                        scaled_keys = keys * scale  # B, H, Lk, D
                        keys_T = scaled_keys.transpose(-1, -2)  # B, H, D, Lk
                        scores = torch.matmul(queries * scale, keys_T)  # B, H, Lq, Lk
                        scores = scores + mask  # B, H, Lq, Lk

                        block_max = torch.amax(scores, dim=-1)  # B, H, Lq sparse
                        running_max = torch.maximum(
                            real_max, block_max
                        )  # B, H, Lq sparse

                        exp_scores = torch.exp(
                            scores - running_max.unsqueeze(-1)
                        )  # B, H, Lq, Lk
                        correction = torch.exp(
                            real_max - running_max
                        )  # B, H, Lq sparse

                        denominator.copy_(
                            denominator * correction + exp_scores.sum(dim=-1)
                        )  # B, H, Lq sparse
                        output.copy_(
                            output * correction.unsqueeze(-1)
                            + torch.matmul(exp_scores, values)
                        )  # B, H, Lq, D

                        real_max.copy_(running_max)  # B, H, Lq sparse

    return output / denominator.unsqueeze(-1)


if __name__ == "__main__":
    torch.manual_seed(SEED)

    queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
    keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
    values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)

    # Causal additive mask in natural [1, 1, Lq, Lk] orientation: query i
    # attends to keys 0..i.  0.0 = keep, -inf = masked.  The kept diagonal
    # guarantees no fully-masked row (no 0/0 NaN denominator).
    causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
    mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
    mask_t.masked_fill_(~causal, float("-inf"))

    queries_t_spyre = queries_t.to(device="spyre")
    keys_t_spyre = keys_t.to(device="spyre")
    values_t_spyre = values_t.to(device="spyre")
    mask_t_spyre = mask_t.to(device="spyre")

    attn_t = flash_cpu(queries_t, keys_t, values_t, mask_t)

    declare_tensor_dim("B", B)
    declare_tensor_dim("H", H)
    declare_tensor_dim("Lq", Lq)
    declare_tensor_dim("Lk", Lk)
    declare_tensor_dim("D", D)

    name_tensor_dims(queries_t_spyre, ["B", "H", "Lq", "D"])
    name_tensor_dims(keys_t_spyre, ["B", "H", "Lk", "D"])
    name_tensor_dims(values_t_spyre, ["B", "H", "Lk", "D"])
    name_tensor_dims(mask_t_spyre, ["B", "H", "Lq", "Lk"])

    c_flash_spyre = torch.compile(flash_spyre)
    attn_t_spyre = c_flash_spyre(
        queries_t_spyre,
        keys_t_spyre,
        values_t_spyre,
        mask_t_spyre,
    )
    torch.testing.assert_close(attn_t, attn_t_spyre.cpu(), atol=0.1, rtol=0.1)
    print("SUCCESS")