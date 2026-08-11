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

"""Real-model programs with multi-dimensional coarse tiling, for the LX-allocation search.

WHY MULTI-DIMENSIONAL. Flash attention is the one workload whose LX allocation is genuinely
hard -- 20 movable buffers, and every shipped solver 13-28 % off optimal
(`research/flash_lx_findings.md`). It gets that space from FOUR nested tiling levels
(`B`, `H`, `Lq`, `Lk`) plus a work division. A program tiled on one dimension produces a
handful of intermediates and a decision anyone can make correctly. So these programs tile
three dimensions each, which is what makes the search space worth searching.

TWO RULES CONSTRAIN WHICH DIMENSIONS CAN BE TILED.

1. `spyre_hint` accepts ONE dimension per call (`propagate_named_dims.py:649`), so
   multi-dimensional tiling is expressed as NESTED `with` blocks, outermost first.
2. Tiling a matmul's REDUCTION dimension makes coarse tiling materialise the read copy over
   the whole iteration space. For `[B,M,K] @ [K,N]` K-tiled by T that is a `[B,M,N,K/T]`
   staging buffer -- 20 MB of inputs became 8.6 GB in the 2026-08-07 sweep, tripped the
   per-core address-span limit, and is the prime suspect in a dead accelerator. **Every
   dimension tiled below is an OUTPUT dimension of every matmul it passes through.**

That second rule is what shapes these programs. A SwiGLU MLP cannot tile `d_ff` across the
down-projection, because `d_ff` is that matmul's reduction axis -- so `mlp_up` stops at the
activation and leaves the down-projection to its caller. Adding a batch dimension supplies a
third safe axis instead.

Every program runs on plain CPU torch, so shapes and numerics are checkable without a device
(`python3 research/workloads.py --check`). The Spyre naming and hints apply only when
torch_spyre imports, so the file stays usable on a machine with no SDK.

Dimensions are Granite's (`tests/resource/models/granite-*.yaml`): `d_model` 4096,
`d_ff` 12800, `head_dim` 128.
"""

import argparse
import contextlib
import math

import torch
import torch.nn.functional as F

try:  # absent on a machine without the SDK, which is fine for --check
    from torch_spyre._inductor import spyre_hint

    try:  # the module moved into the `wsr` package on the torch 2.13 line
        from torch_spyre._inductor.wsr.propagate_named_dims import (
            declare_tensor_dim,
            name_tensor_dims,
        )
    except ImportError:  # torch 2.11 line (`dev1`) keeps it flat
        from torch_spyre._inductor.propagate_named_dims import (
            declare_tensor_dim,
            name_tensor_dims,
        )

    HAVE_SPYRE = True
except Exception:  # noqa: BLE001 -- any import failure means "no SDK here"
    HAVE_SPYRE = False

    def spyre_hint(**_kw):  # type: ignore[misc]
        return contextlib.nullcontext()

    def declare_tensor_dim(*_a):  # type: ignore[misc]
        return None

    def name_tensor_dims(t, _names):  # type: ignore[misc]
        return t


GRANITE = {"d_model": 4096, "d_ff": 12800, "head_dim": 128, "n_heads": 32, "eps": 1e-5}


def _tiled(**counts):
    """Nest one `spyre_hint` per dimension, outermost first.

    `spyre_hint` rejects a dict naming more than one dimension, so a three-level tiling is
    three nested context managers. Levels with a count of 1 are dropped -- the hint pass
    discards them anyway (`coarse_tile_hints.py:66`) and they only clutter the IR.
    """
    stack = contextlib.ExitStack()
    for dim, n in counts.items():
        if n and n > 1:
            stack.enter_context(spyre_hint(num_tiles_per_dim={dim: n}))
    return stack


def _declare(**dims):
    for name, size in dims.items():
        declare_tensor_dim(name, size)


def mlp_up(
    batch=4, seq=512, d_model=None, d_ff=None, bt=2, st=4, ft=4, dtype=torch.float16
):
    """Batched SwiGLU up-projection and activation, tiled on B x S x F.

    `silu(x @ Wg) * (x @ Wu)` over `x[B, S, D]`. The two projections share `x`, so their
    outputs `gate` and `up` are simultaneously live across the activation -- the working set
    that makes LX interesting. All three tiled dimensions (`B`, `S`, `F`) are OUTPUT
    dimensions of both matmuls; the reduction axis `D` is untouched.

    The down-projection is deliberately NOT here. It reduces over `F`, so tiling `F` would
    split its reduction and trigger the read-copy amplification described in the module
    docstring. Its caller applies it outside the tiled region.
    """
    d_model = d_model or GRANITE["d_model"]
    d_ff = d_ff or GRANITE["d_ff"]

    def build():
        return (
            torch.randn(batch, seq, d_model, dtype=dtype),
            torch.randn(d_model, d_ff, dtype=dtype) * 0.02,
            torch.randn(d_model, d_ff, dtype=dtype) * 0.02,
        )

    def fn(x, wg, wu):
        _declare(B=batch, S=seq, D=d_model, Fd=d_ff)
        name_tensor_dims(x, ["B", "S", "D"])
        name_tensor_dims(wg, ["D", "Fd"])
        name_tensor_dims(wu, ["D", "Fd"])
        with _tiled(B=bt, S=st, Fd=ft):
            gate = x @ wg
            up = x @ wu
            return F.silu(gate) * up

    return fn, build


def attn_scores(
    batch=2,
    heads=8,
    seq_q=512,
    seq_k=512,
    head_dim=None,
    bt=2,
    ht=4,
    qt=4,
    dtype=torch.float16,
):
    """Attention through the softmax, tiled on B x H x Lq, with an explicit transpose.

    `softmax(q · kᵀ · scale) @ v`. The transpose materialises as a `clone` carrying the
    `restickify` access pattern, which the cost model prices at ~116 GB/s against the default
    150 -- so this program puts buffers of DIFFERENT effective bandwidth in contention for
    the same scratchpad. That heterogeneity is the precondition for a time-based ranking to
    differ from a byte-based one at all (`findings_lx.md` section 6).

    `Lk` is not tiled: it is the reduction axis of both the score matmul and the `probs @ v`
    matmul. Flash attention tiles it only because its online-softmax recurrence carries the
    running max and denominator across tiles; this program has no such recurrence.
    """
    head_dim = head_dim or GRANITE["head_dim"]
    scale = 1.0 / math.sqrt(head_dim)

    def build():
        return (
            torch.randn(batch, heads, seq_q, head_dim, dtype=dtype),
            torch.randn(batch, heads, seq_k, head_dim, dtype=dtype),
            torch.randn(batch, heads, seq_k, head_dim, dtype=dtype),
        )

    def fn(q, k, v):
        _declare(B=batch, H=heads, Lq=seq_q, Lk=seq_k, Dh=head_dim)
        name_tensor_dims(q, ["B", "H", "Lq", "Dh"])
        name_tensor_dims(k, ["B", "H", "Lk", "Dh"])
        name_tensor_dims(v, ["B", "H", "Lk", "Dh"])
        with _tiled(B=bt, H=ht, Lq=qt):
            kt = k.transpose(-1, -2).contiguous()  # -> restickify buffer
            scores = torch.matmul(q * scale, kt)
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)

    return fn, build


def block_norm_mlp(
    batch=4, seq=512, d_model=None, d_ff=None, bt=2, st=4, ft=4, dtype=torch.float16
):
    """RMSNorm then the SwiGLU up-projection, tiled on B x S x F.

    The one program here that mixes access-pattern kinds: a cross-row reduction (the norm's
    `mean(x*x)`), a broadcast (the reciprocal scale applied back across `D`), and full-width
    matmul operands, all competing for the same scratchpad in one bundle. The norm's
    intermediates are tiny and its reduction result is reused by every column, so the
    residency trade-off here is between many small high-reuse buffers and few large ones --
    a different shape of decision from `mlp_up`.

    `D` is not tiled: it is the norm's reduction axis and both projections' reduction axis.
    """
    d_model = d_model or GRANITE["d_model"]
    d_ff = d_ff or GRANITE["d_ff"]
    eps = GRANITE["eps"]

    def build():
        return (
            torch.randn(batch, seq, d_model, dtype=dtype),
            torch.randn(d_model, dtype=dtype),
            torch.randn(d_model, d_ff, dtype=dtype) * 0.02,
            torch.randn(d_model, d_ff, dtype=dtype) * 0.02,
        )

    def fn(x, w, wg, wu):
        _declare(B=batch, S=seq, D=d_model, Fd=d_ff)
        name_tensor_dims(x, ["B", "S", "D"])
        name_tensor_dims(wg, ["D", "Fd"])
        name_tensor_dims(wu, ["D", "Fd"])
        with _tiled(B=bt, S=st, Fd=ft):
            ms = (x * x).mean(dim=-1, keepdim=True)
            xn = x * torch.rsqrt(ms + eps) * w
            gate = xn @ wg
            up = xn @ wu
            return F.silu(gate) * up

    return fn, build


def decode_block(
    batch=2,
    heads=8,
    cache_len=1024,
    new_len=128,
    head_dim=None,
    d_model=None,
    d_ff=None,
    bt=2,
    ht=4,
    ft=4,
    dtype=torch.float16,
):
    """Decode-step transformer block with a KV-cache append -- built to break CP-SAT.

    WHY THIS PROGRAM EXISTS. CP-SAT maximises retained BYTES; the cost model ranks by TIME.
    They differ only when contending buffers move at different bandwidths, and then only when
    the SLOW buffer is the SMALLER one -- otherwise both objectives want the same buffer and
    agree. Precisely, with A slow and B fast, a search beats CP-SAT when

        bw_A / bw_B  <  bytes_A / bytes_B  <  1.

    Flash attention has the bandwidth spread (its `cat` outputs run at 105-118 GB/s against
    150) but fails the size test: those outputs are its LARGEST buffers, so CP-SAT keeps them
    anyway and is exactly time-optimal. This program supplies both halves at once:

    * `k_all`/`v_all` -- a `cat` on the sequence (partition) dim, which the extractor tags
      `stick_scatter` (`dump_cost_model.py:442`) and the model prices by
      `clamp(144 - 9.6*log2(C/64) - 2.4*log2(R), 44, 150)`. At Granite width that is at or
      near the **44 GB/s floor, 3.41x below default**, opening the window to bytes ratios
      as low as 0.29.
    * `gate`/`up` -- the SwiGLU intermediates at `d_ff` = 12800, `default` pattern at
      150 GB/s and several times larger than the cache tensors.

    So the slow buffers are the medium ones and the fast buffers are the big ones, which is
    the ordering CP-SAT cannot see. Whether the byte ratio actually lands inside the window
    depends on `cache_len` vs `d_ff`, which is what `screen_configs.py` sweeps.

    TILED DIMENSIONS. `B`, `H`, `Fd` -- every one an OUTPUT dim of every matmul it crosses.
    `D` is the reduction axis of the QKV and MLP projections, `Dh` of the output projection,
    and `cache_len + new_len` of `probs @ v_all`; none of them is tiled, per the read-copy
    rule in the module docstring.
    """
    d_model = d_model or GRANITE["d_model"]
    d_ff = d_ff or GRANITE["d_ff"]
    head_dim = head_dim or GRANITE["head_dim"]
    eps = GRANITE["eps"]
    scale = 1.0 / math.sqrt(head_dim)

    def build():
        return (
            torch.randn(batch, heads, new_len, d_model, dtype=dtype),
            torch.randn(batch, heads, cache_len, head_dim, dtype=dtype),
            torch.randn(batch, heads, cache_len, head_dim, dtype=dtype),
            torch.randn(d_model, head_dim, dtype=dtype) * 0.02,
            torch.randn(d_model, head_dim, dtype=dtype) * 0.02,
            torch.randn(d_model, head_dim, dtype=dtype) * 0.02,
            torch.randn(head_dim, d_model, dtype=dtype) * 0.02,
            torch.randn(d_model, d_ff, dtype=dtype) * 0.02,
            torch.randn(d_model, d_ff, dtype=dtype) * 0.02,
        )

    def fn(xh, k_cache, v_cache, wq, wk, wv, wo, wg, wu):
        _declare(
            B=batch, H=heads, Sn=new_len, Sc=cache_len, D=d_model, Dh=head_dim, Fd=d_ff
        )
        name_tensor_dims(xh, ["B", "H", "Sn", "D"])
        name_tensor_dims(k_cache, ["B", "H", "Sc", "Dh"])
        name_tensor_dims(v_cache, ["B", "H", "Sc", "Dh"])
        with _tiled(B=bt, H=ht, Fd=ft):
            q = xh @ wq
            k_new = xh @ wk
            v_new = xh @ wv
            # The cat0: concatenating along the SEQUENCE dim wedges a small device dim
            # just inside the 64-stick, which is the tagged stick_scatter path.
            k_all = torch.cat([k_cache, k_new], dim=-2)
            v_all = torch.cat([v_cache, v_new], dim=-2)
            scores = torch.matmul(q * scale, k_all.transpose(-1, -2))
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, v_all)
            attn = ctx @ wo
            ms = (attn * attn).mean(dim=-1, keepdim=True)
            xn = attn * torch.rsqrt(ms + eps)
            return F.silu(xn @ wg) * (xn @ wu)

    return fn, build


def prefix_block(
    batch=4,
    prefix_len=512,
    new_len=512,
    d_model=None,
    d_ff=None,
    bt=2,
    st=4,
    ft=4,
    dtype=torch.float16,
):
    """Chunked-prefill block: concat hidden states on the SEQ dim, then RMSNorm + SwiGLU.

    THE ONE PROGRAM HERE WHERE A TIME-RANKED SEARCH BEATS CP-SAT. The disagreement needs a
    buffer that is SLOW but CHEAP by the byte objective, losing to one that is FAST but
    EXPENSIVE. Both halves are arranged deliberately:

    * `h = cat([h_prefix, h_new], dim=seq)` is a cat on a partition dim -> `stick_scatter`,
      and its stick dim is `d_model` = 4096, so `clamp(144 - 9.6*log2(C/64) - 2.4*log2(R),
      44, 150)` gives **~62 GB/s**. It has TWO consumers (the norm's `mean` and the rescale),
      so CP-SAT scores it `3 * 4096 = 12288`.
    * `gate`/`up` are `d_ff` = 12800 wide at the default 150 GB/s with ONE consumer each, so
      CP-SAT scores them `2 * 12800 = 25600` -- twice `h`.

    CP-SAT therefore keeps `gate` and spills `h`. In time that is backwards: spilling `h`
    costs `3*4096/62 = 198` against `gate`'s `2*12800/150 = 171`. The byte objective cannot
    see it, because the ratio it is blind to -- 2.4x in bandwidth -- is exactly what inverts
    the ranking.

    Contrast `decode_block`, which has a cat but on the KV cache, whose stick dim is
    `head_dim` = 128. That yields ~110 GB/s, a 1.36x spread, too little to overturn anything
    -- and the screen duly reports zero wins there. The stick WIDTH of the concat is the
    variable that matters, which is not obvious until the formula is written down.

    `D` is not tiled (the reduction axis of both projections). `ft` shrinks `gate`/`up` and
    leaves `h` alone, so sweeping it sweeps the byte ratio across the disagreement window.
    """
    d_model = d_model or GRANITE["d_model"]
    d_ff = d_ff or GRANITE["d_ff"]
    eps = GRANITE["eps"]

    def build():
        return (
            torch.randn(batch, prefix_len, d_model, dtype=dtype),
            torch.randn(batch, new_len, d_model, dtype=dtype),
            torch.randn(d_model, d_ff, dtype=dtype) * 0.02,
            torch.randn(d_model, d_ff, dtype=dtype) * 0.02,
        )

    def fn(h_prefix, h_new, wg, wu):
        _declare(
            B=batch,
            Sp=prefix_len,
            Sn=new_len,
            S=prefix_len + new_len,
            D=d_model,
            Fd=d_ff,
        )
        name_tensor_dims(h_prefix, ["B", "Sp", "D"])
        name_tensor_dims(h_new, ["B", "Sn", "D"])
        name_tensor_dims(wg, ["D", "Fd"])
        name_tensor_dims(wu, ["D", "Fd"])
        with _tiled(B=bt, S=st, Fd=ft):
            h = torch.cat(
                [h_prefix, h_new], dim=-2
            )  # cat0 -> stick_scatter, C = d_model
            ms = (h * h).mean(dim=-1, keepdim=True)
            xn = h * torch.rsqrt(ms + eps)
            return F.silu(xn @ wg) * (xn @ wu)

    return fn, build


#: name -> factory. Each takes bt/st/ft (or bt/ht/qt) tile counts as its knobs, the same
#: shape of control flash attention exposes through FA_*_TILES.
WORKLOADS = {
    "mlp_up": mlp_up,
    "attn_scores": attn_scores,
    "block_norm_mlp": block_norm_mlp,
    "decode_block": decode_block,
    "prefix_block": prefix_block,
}

#: Tile-count grids to search. Every count must divide its dimension exactly
#: (`coarse_tile.py:855` raises otherwise), and the product bounds the loop nest depth.
KNOB_GRID = {
    "mlp_up": {"bt": (1, 2, 4), "st": (1, 2, 4, 8), "ft": (1, 2, 4, 8)},
    "attn_scores": {"bt": (1, 2), "ht": (1, 2, 4, 8), "qt": (1, 2, 4, 8)},
    "block_norm_mlp": {"bt": (1, 2, 4), "st": (1, 2, 4, 8), "ft": (1, 2, 4, 8)},
    "decode_block": {"bt": (1, 2), "ht": (1, 2, 4, 8), "ft": (1, 2, 4, 8)},
    "prefix_block": {"bt": (1, 2, 4), "st": (1, 2, 4, 8), "ft": (1, 2, 4, 8)},
}


def _check():
    """Run every workload on CPU at several knob settings — no Spyre device involved."""
    print(f"torch_spyre available: {HAVE_SPYRE}\n")
    ok = True
    for name, factory in WORKLOADS.items():
        grid = KNOB_GRID[name]
        knobs = {k: v[len(v) // 2] for k, v in grid.items()}
        try:
            fn, build = factory(**knobs)
            args = build()
            out = fn(*args)
            finite = bool(torch.isfinite(out.float()).all())
            ok &= finite
            print(
                f"  {name:<16} knobs={knobs} in={tuple(args[0].shape)} "
                f"out={tuple(out.shape)} finite={finite}"
            )
        except Exception as exc:  # noqa: BLE001 -- report, do not abort the sweep
            ok = False
            print(f"  {name:<16} FAILED: {type(exc).__name__}: {exc}")
    print("\nall workloads well-formed" if ok else "\nSOME WORKLOADS FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="run all workloads on CPU")
    ap.add_argument("--grid", action="store_true", help="print the knob search space")
    args = ap.parse_args()
    if args.check:
        return _check()
    if args.grid:
        total = 0
        for name, grid in KNOB_GRID.items():
            n = 1
            for v in grid.values():
                n *= len(v)
            total += n
            dims = " x ".join(f"{k}{list(v)}" for k, v in grid.items())
            print(f"  {name:<16} {n:>4} settings   {dims}")
        print(f"\n  {total} configurations in the search space")
        return 0
    print("workloads:", ", ".join(sorted(WORKLOADS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
