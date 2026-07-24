# Part III rewrite DRAFT — matmul compute/HBM overlap (cats 3+4+6)

**STATUS: DRAFT, not merged.** This is the report-ready prose for the coordinated Part III
rework (the cat-3 overlap mechanism + operand-aware spill + cat-4 bmm + cat-6 coarse). It is
kept OUT of `cost_model_report.md` until the model actually ships, so the report keeps tracking
the *shipped* model (γ=0.46, peak 1140). Merge §9/§10/§11 below when the code change lands;
FINALIZE the numeric coefficients (peak, γ, spill) on the clean forced-core sweep data
(`run_matmul_overlap_iso_sweep.sh` + `run_coarse_bwcores_sweep.sh`) — the values here are from the
noisy overnight data and are provisional. Full spec + validation in `cost_model_status.md`.

---

## §9 (revised). The compute term — a measured sustained rate, not the datasheet peak

The per-core compute time is `MACs / cores / (mac_peak · pt_eff)`. The load-bearing constant is
`mac_peak`, the sustained MAC rate one core drives into the PT array. Forcing a matmul onto a
*single* core makes it purely compute-bound (its HBM floor is 2–5 % of the kernel time), so the
measured `MACs / (cores · time)` reads the rate directly, with no overlap or memory to untangle.
Across every compute-dominant run the implied rate is **~1046 MAC/ns/core** (tight — a few percent
spread over dozens of shapes), well below the 1536 datasheet peak. This is the number to use.

Why this matters beyond accuracy: the previously-shipped model used 1140 here, ~9 % high. A
9 %-inflated compute term does not just shift predictions — it forces the *overlap* term to absorb
the error, and because the amount it must absorb depends on the compute/memory balance, that made
a single overlap constant look as if it varied with shape. Getting the rate right is what lets the
overlap be a clean constant (next section).

## §10 (revised). Compute and HBM overlap — compute-bounded double-buffering

**Observation.** Adding compute and memory over-predicts: the real kernel is faster than the sum,
because the accelerator streams the *next* tile's operands from memory while the array computes the
current one (double-buffering). But the two do not simply run concurrently — three facts pin the
shape of the overlap, all measured:

1. **Only the loads hide, not the stores.** Operand *reads* can be prefetched under compute; the
   output *write* happens after its values are computed, so it cannot hide behind the same kernel's
   compute. Charging writes serially (and hiding only reads) fits markedly better than hiding all
   memory — most visibly on write-heavy shapes (wide-N, thin-K), which the all-memory form
   over-optimistically predicts.
2. **The hidden amount is capped by the compute duration.** Loads hide only while the array is
   running, so at most a fixed fraction γ of the compute time is an overlap window. When the reads
   are shorter than that window they hide completely; when they are longer, only γ·compute hides.
3. **γ is a constant.** It is the double-buffer window fraction, a property of the pipeline, not of
   the shape. (An earlier analysis appeared to show γ rising with tile aspect ratio and falling with
   write/read ratio; that turned out to be measurement noise on un-repeated points — the trend
   vanished on repeat-backed data. See the methodology note on back-out noise.)

**Model.**

```
T = compute + read + write + turn − min(read, γ·compute)
    read  = (operand_bytes + spill) / bw_read     (loads — double-buffered)
    write = output_bytes / bw_write                (stores — serial)
    γ ≈ 0.6   (double-buffer window fraction, constant)
```

For a non-matmul bundle compute = 0, so `min(read, 0) = 0` and the term vanishes — pointwise,
reduction and transport ops are untouched. The effective overlap *appears* to vary with shape —
a compute-heavy tile hides all its reads (`T → compute + write`), a memory-heavy tile hides only
γ·compute — but that variation is a consequence of `min(read, γ·compute)` switching regimes, with a
single constant γ underneath. That is the resolution of "γ cannot be constant": it is constant; the
*net* overlap is access-pattern-dependent by construction.

## §11 (revised). Operand-aware spill — re-read is bounded by the smaller operand

When the per-core output tile `(M/m)·(N/n)` overflows on-chip capacity, an operand must be
re-streamed from memory. The re-read is **not** symmetric in the two operands: the compiler keeps
the larger operand resident and re-streams the smaller one, so the extra traffic scales with
`2·min(|A|, |B|)`, not `(|A|+|B|)`. The old symmetric form over-charged tall-operand / thin-N tiles
(where |A| ≫ |B|) and under-charged wide-N tiles — a bidirectional residual that the min form
removes. The re-read *fraction* still grows with how far the tile overflows the capacity knee
(`f(area)` unchanged), and the re-read is charged to the read side, so it is itself double-buffered.

**Note on entanglement (why this ships as one change).** These three revisions are coupled: the
operand-aware spill *hurts* under the old `γ·min(compute,mem)` overlap form and *helps* under the
read-overlap form; raising the peak shifts what the spill and split terms must absorb. They were
originally co-fit, so they must be re-fit together. The bmm layout rate (cat 4) and the coarse
per-tile fill/drain (cat 6, the same double-buffering physics applied per coarse tile) build on this
same compute model and land in the same coordinated change.
