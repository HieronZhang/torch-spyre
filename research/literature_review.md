# Compiler Research for Dataflow Accelerators: A 2024–2026 Literature Review

*Prepared for the torch-spyre project (IBM Spyre / AIU PyTorch backend).*
*Covers ten topics and roughly 225 sources, peer-reviewed and industrial. Preference
for 2024–2026 material; older work cited only where canonical. Claims I could not
verify are flagged inline and collected in §15; every source is indexed in §16.*

---

**Revision note.** This is a full revision of the first draft, restructured from
thirteen sections to sixteen and expanded by four topics. Readers of the draft should
know exactly what changed, because several of the changes are corrections rather than
additions.

*Errors corrected.* (1) Roller was described as an OSDI'22 best paper. It was not —
the 2022 Jay Lepreau awards went to MemLiner, XRP and Sieve — and the claim is gone;
the argument never needed it. (2) The draft asserted "there is no runtime model in
Inductor's fusion path." That is too strong. `speedup_by_fusion`'s multi-template path
does call `_get_estimated_runtime()` and feed it to `_estimate_fused_epilogue_runtime`,
so exactly one runtime-estimate-driven fusion decision exists upstream; it is gated on
`MultiTemplateBuffer` plus Triton templates, which is why it can never fire for a
PrivateUse1 backend. §3.2 now states it that way. (3) The draft claimed a PrivateUse1
backend "inherits no timing model at all" and that "no upstream machinery will consume
it unaided." Wrong on both counts in the narrow sense: `config.estimate_op_runtime` is
a user-supplied callable dispatched in `comms.py`, and `analysis/device_info.py` is a
datasheet table a custom device could register in. The accurate statement — one hook
exists, wired only to comms reordering — is §3.3. (4) The draft claimed nobody had
rank-correlated a model against measured latency behind a proprietary backend.
TpuGraphs (NeurIPS'23 D&B) and Kaufman et al. (MLSys'21) do exactly that with *learned*
models against measured TPU latency behind XLA:TPU. The surviving novelty is narrower
and is stated as such throughout: an *analytical* model, scored by rank, on an
explicit-NoC dataflow chip. (5) The draft claimed nobody co-optimises tiling and
scratchpad allocation. That was the single largest factual error: Welder, Stream,
DeFiNES, TileFlow, Elk, CoSA, AKG and Baskaran et al. all do, and §7 is largely a
rewrite around them. The related claim that "no published system spills within a fused
kernel" is softened to what actually survives — nobody prices the cost *gradient* near
capacity, and nobody feeds allocator outcome back into tile selection. (6) The draft
opened with "the unit of compilation is now the tile, and that argument is over." XLA,
IREE, AKG and Mosaic all still compile from structured loop IR with compiler-chosen
tiling, and MLIR's Linalg plus `transform` dialect is a live competing answer to who
owns the schedule; the claim is now scoped to GPU DSLs. (7) Axe's Trainium/NKI result,
which two draft sections leaned on, could not be confirmed from the abstract and has
moved to §15. (8) Unverified PyTorch release facts — the 2.12 ship date,
`torch.accelerator.Graph`, FlashAttention-4 in 2.11 — have moved to §15. Smaller
corrections carried in the same pass: Buffets is ASPLOS 2019, not ISCA 2019; Diesel is
a MAPL'18 workshop paper and no number from it should be cited; DNNFusion's 9.3× is a
maximum against the weakest baseline; MAESTRO publishes two non-interchangeable
accuracy claims; Habitat's 11.8% is end-to-end iteration error, not per-operation.

*Sections added.* §2 (MLIR ecosystem and the tile-IR decision) — the draft cited only
tt-mlir and CUDA Tile IR and was therefore missing the main open competitor to a
bespoke tile IR. §6 now leads with IBM's own directly analogous compiler, onnx-mlir's
NNPA backend, which is where the word *stick* comes from and which already ships a
measurement-fitted analytical cost model for stick hardware; the draft cited neither it
nor zDNN. §7 is rebuilt around joint tiling and residency. §8.4 (in the tail) adds the
latency-prediction methodology the draft skipped. §9 (bucketing and the serving
objective) and §10 (block-scaled quantization as a layout problem) are new topics the
draft treated as out of scope and should not have. §14 now states what remains out of
scope and why, rather than leaving it implicit.

---

## 0. Executive summary

**The tile won as a language value — in GPU DSLs, and that scope matters.** Triton's
2019 contribution was to make a statically-shaped block a *value* rather than a
directive attached to a loop, and every serious GPU DSL since has accepted the
resulting contract: the programmer owns the grid, the blocking, the loop order and the
masking; the compiler owns everything inside the block. Gluon, Hexcute and the CuTe DSL
push the line *down* toward more user control; Helion pushes it *up*; NVIDIA's CUDA
Tile IR moves it *into the vendor stack*. But outside that world the argument is not
over. XLA, IREE, AKG and Mosaic all still compile from structured loop IR with
compiler-chosen tiling, and MLIR's Linalg plus `transform` dialect is a live, shipping
answer to the same question that keeps the schedule in the compiler and makes it
searchable data instead. There is no consensus tile IR in 2026, and a third-party
evaluation suggests a tile IR buys code size rather than performance portability.

**The deepest technical result of the period is that layout became an algebra.**
Triton's ad-hoc layout enum produced a quadratic explosion of conversion paths; the
Linear Layouts work replaced it with binary matrices over F₂ acting on the *bits* of
(thread, register, tensor) indices. The headline argument is correctness, not speed:
mixed-precision matmul pass rate went from 46.6% to 100%, against a backdrop where 12%
of filed Triton bugs are layout-related. CuTe's hierarchical (shape, stride) layouts
are the same idea; Hexcute *synthesizes* them by constraint propagation anchored at
GEMM operators; Axe generalizes to set-valued maps over named hardware axes and drops
the power-of-two restriction. IBM's own DNNDaSher makes the point from the hardware
side and states it more strongly: on AIU, producer/consumer element-organization
mismatch is a *correctness* constraint, and systematically eliminating and coarsening
the resulting shuffles was worth 1.27×–4.12× (avg 2.3×) end-to-end.

**IBM already shipped the ancestor of half this problem, and the draft missed it.**
onnx-mlir's NNPA backend for the IBM Z Integrated Accelerator lowers ONNX through
**ZHigh** and **ZLow** with explicit `stickify`/`unstickify` operations over zDNN's
stick layout — the origin of the term torch-spyre uses. It ships conversion-count
benefit tests, conversion/compute fusion, compile-time stickification of constants, a
placement heuristic that charges stick/unstick per graph edge inside a five-round fixed
point, and `PerfModelArch14/15.inc`: per-op regressions fitted against measured z16/z17
hardware, with r² recorded in the source. Its worst fits are Unstick (0.429), Stick
(0.691) and MatMul_3ds (0.706). So torch-spyre is not the first analytical cost model
for stick hardware, and the hardest terms to fit there are precisely the layout
conversions. That is simultaneously a caution and a reassurance.

**"Can a compiler discover FlashAttention?" now has a qualified yes, and the
qualification is the interesting part.** Mirage rediscovers FlashAttention and
FlashDecoding hint-free by superoptimizing over µGraphs. Neptune *derives* the
online-softmax correction term by symbolic solving rather than pattern matching.
Flashlight does template-free attention fusion inside TorchInductor. Nautilus produces
FA3-like kernels from a math-level description in under a minute of scheduling. But the
flagship production effort went the other way: PyTorch reports FlexAttention fell from
~80% to ~60% of FlashAttention-3 throughput, and their fix was not a better compiler
but making FlexAttention a *frontend* that JIT-instantiates the hand-written FA4 kernel.
And for a tiled dataflow chip there is a further wrinkle: FlatAttention reports that the
FlashAttention *dataflow itself* is the wrong target, and that rebuilding attention
around on-chip NoC collectives gives 4.1× speedup and 16× less HBM traffic on the same
hardware.

**Hand-written kernels win because they operate on a different program representation,
not because humans pick better tile sizes.** FlashMLA's schedule was forced by a
register-file capacity constraint; DeepGEMM's block N=112 is a wave-quantization fix,
not a locality fix; QuACK beats `torch.compile` on softmax by 1.6× purely because
Inductor emits one extra global load. The encouraging counter-evidence is Twill, which
formulates joint software pipelining and warp specialization as ILP+SMT and
*rediscovers* the FA3 schedule to within 1% — and the Triton warp-specialization
roadmap, which names "model-based global optimization" using static or profiled cost
models as its stated future direction.

**Upstream TorchInductor is less than people assume, but not as little as the draft
said.** Fusion is ranked by a four-element lexicographic tuple whose only substantive
term is summed bytes of shared memory dependencies. Exactly one runtime-estimate-driven
fusion decision exists — in `speedup_by_fusion`'s multi-template path — and it is gated
on `MultiTemplateBuffer` plus Triton templates, so it never fires for a PrivateUse1
backend. The underlying model, `_get_estimated_runtime`, is a roofline with
`factor = 1.0`, a TODO admitting inadequacy, and an early `return 0` for any non-GPU
device. There *is* one user-supplied cost-model hook, `config.estimate_op_runtime`, and
it is wired exclusively to communication-overlap reordering; and there *is* a datasheet
table, `analysis/device_info.py`, where a custom device could register its TOPS and
bandwidth so the roofline stops returning zero. Real memory planning exists and is
default-off. Outside Inductor the general question is already answered: XLA:GPU's
PriorityFusion ranks fusion candidates by predicted runtime deltas in production.

**On-chip memory: the draft's central negative claim was wrong, and what survives is
sharper.** Two literatures still look disjoint at first glance — the packing line
(TelaMalloc, MiniMalloc) treats allocation as pure feasibility with no cost model
anywhere, and the mapping line (Timeloop, CoSA, TileLoom) treats capacity as a validity
predicate that filters candidates before the cost model ranks survivors. But *joint*
tiling and on-chip residency is well established: Welder schedules tile-graph data
movement across memory layers with a tile-traffic cost model and checks footprint
against level capacity inside the search; Stream puts per-core capacity directly in its
ILP; DeFiNES sweeps tile size against capacity for fused layer stacks; TileFlow models
384 KB per core across four cores; Elk partitions each core's SRAM between execution
and preload space; and the polyhedral line (Baskaran et al., PPCG, AKG) has done tile
size plus scratchpad promotion as one affine problem since 2008. What none of them do
is *price* capacity: Welder assigns infinite penalty above it, DeFiNES picks the lowest
level that fits, Stream writes a hard inequality, PPCG refuses promotion, MLIR recurses
inward. Nobody models the cost gradient near capacity, and nobody feeds allocator
outcome back into tile selection.

**Tile-size search splits into three camps, and only one fits a remote, opaque
backend.** Measurement-guided search (Ansor, Helion) assumes cheap on-device
evaluation. Construction (Roller, Hidet, Bolt, Heron) collapses the space by hardware
alignment or by constraints so a simple static model suffices — and Roller's decisive
result for this project is on Graphcore IPU, an immature accelerator with a slow opaque
device compiler, where it beat the vendor library by 3.1× average. Analytical *ranking*
has now been validated at scale on GPUs: tritonBLAS reports 94.7% selection efficiency
against exhaustive measurement over 150,000 GEMM shapes; TileSight's top-5% prune
retains 99.66% of exhaustive best. On scoring, TenSet is decisive and TLP, nn-Meter and
Habitat supply the conventions: rank and top-k, not RMSE; report the ±10% fraction;
separate per-op error from importance-weighted end-to-end error.

**Two topics the draft excluded belong in scope.** Static shapes do not make bucketing
irrelevant — they make *choosing the bucket and pad set* a compiler cost-model decision
with a serving SLO attached, and DietCode's multiplicative model with its core-occupancy
factor is startlingly close to a 32-core ragged edge. And FP8 is not a narrower dtype
that leaves stick alignment alone: block-scaled formats put a sub-stick-granular scale
tensor, with no free transpose, inside the layout solver.

**The convergent gap is where torch-spyre stands.** Tenstorrent's tt-mlir optimizer spec
describes an unimplemented "Cost Mode" that would replace its heuristic layout score
with real runtime estimates. The Triton warp-specialization roadmap names cost-model-
driven joint search as future work. TileLoom is the existence proof that this works on
real spatial dataflow silicon — 17% geomean prediction error was enough to rank
correctly, and top-2 profiling added only +4.7%. What is unclaimed is narrower than the
draft said and still worth claiming: an analytical model, scored by rank, over a joint
tile × work-division space, on a per-core-partitioned scratchpad machine behind a
proprietary backend, with the allocator's outcome priced rather than filtered.

---

## 1. Tile abstractions and layout algebras

### 1.1 The lineage, compressed

[Halide](https://andrew.adams.pub/halide_cacm.pdf) (CACM 2018, canonical write-up of
the 2012–13 work) established the founding invariant: separating the algorithm from
the schedule means changing the schedule can only change performance, never output.
TVM industrialized it for tensors; [Ansor](https://arxiv.org/abs/2006.06762)
(OSDI'20) automated schedule search over a hierarchical sketch space with a learned
cost model, reporting up to 3.8×/2.6×/1.7× over prior state of the art on Intel CPU,
ARM CPU and NVIDIA GPU. That branch stalled for two reasons: search cost measured in
hours, and a schedule vocabulary of loop transforms on scalar loop nests that could
not express warp-specialized asynchronous pipelines.

[Triton](https://dl.acm.org/doi/abs/10.1145/3315508.3329973) (MAPL@PLDI 2019) changed
the primitive rather than the search. Three subsequent redraws of its compiler/user
boundary matter. **Downward:**
[Gluon](https://triton-lang.org/main/gluon/index.html) shares Triton's stack but
deliberately re-exposes layouts, shared memory and warp specialization;
[Hexcute](https://arxiv.org/html/2504.16214) (CGO 2026) targets kernel engineers,
exposing shared memory and registers while *inferring* CuTe layouts; the
[CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
brings the full C++ layout algebra, atoms and thread-data hierarchy into Python.
**Upward:** [Helion](https://pytorch.org/blog/helion/) is "PyTorch with tiles," where
one `hl.tile` implicitly defines a search space of block sizes, loop orders, PID
mappings, indexing modes and pipeline depths. **Sideways:** NVIDIA's
[CUDA Tile IR](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
(CUDA 13.1, Apache-2.0 since Dec 2025) makes tile-level the canonical MLIR-based IR,
and the Triton-to-TileIR backend preserves tile semantics instead of lowering to
SIMT/PTX. See also NVIDIA's framing post,
["Focus on Your Algorithm"](https://developer.nvidia.com/blog/focus-on-your-algorithm-nvidia-cuda-tile-handles-the-hardware/),
which describes Tile IR as a sibling to PTX at array rather than thread granularity.

The scope of that story is GPU DSLs. §2 covers the parallel line — Linalg, the
`transform` dialect, IREE and Mosaic — where the compiler still chooses the tiling and
the schedule is expressed as searchable IR rather than as user-written tile code. Both
camps are live in 2026, and a backend choosing an IR is choosing between them.

### 1.2 What actually won — and what transfers

Three abstractions won, and they are separable, which matters because a dataflow
backend can adopt some and must reject others.

| Won abstraction | Canonical instances | Transfers to Spyre? |
|---|---|---|
| Tile as a language value | Triton, TileLang, Hexcute, ThunderKittens, Pallas, Helion, CuTile | Yes — this is what KTIR is |
| Layout as an algebra | Linear Layouts (F₂), CuTe, Axe, Graphene | Yes in principle; no published Spyre instance |
| Async pipeline as schedule annotation | `T.Pipelined`, `plgpu` warp-specialized pipelines, Helion `num_stages` | **No** — the overlap lives in DeepTools |
| Thread–value map underneath all three | TV-layouts, warp-collective atoms, F₂ bits of (thread, register) | **No** — Spyre has no threads |

[TileLang](https://arxiv.org/abs/2504.17577) is the clearest statement of the
abstraction to imitate: decoupling the scheduling space (thread binding, layout,
tensorize, pipeline) from the dataflow as annotations, reaching 1.36× over FA3 and
1.41× over Triton on H100 attention, MLA at 98% of hand-written FlashMLA in ~70 lines,
and GEMM at 0.97–1.10× of cuBLAS/rocBLAS across 4090/A100/H100/MI300X. Its layout
inference is anchored on GEMM tensor-core atoms and thread bindings, so only the
dataflow/schedule split transfers.

On layouts, [Linear Layouts](https://arxiv.org/abs/2505.23819) is the empirical case
for replacing an enum-ish layout descriptor scheme with a composable algebra: layout
conversions up to 3.93×, gathers up to 14.20×, 265 TritonBench cases averaging 1.07×,
and the pass-rate and bug-rate figures quoted in §0.
[Axe](https://arxiv.org/pdf/2601.19092) (2026) is the strongest candidate for what
comes next — a layout as a set-valued map `L(x) = {D(x) + r + O}` over *named hardware
axes*, with sharding, replication and offset first-class, deliberately dropping F₂'s
power-of-two restriction. It explicitly does *not* yet address dataflow/streaming
accelerators. The draft additionally credited Axe with a Trainium/NKI result — matching
hand-written NKI on GEMM, beating it up to 1.44× on MHA, 228 lines against 1188 — and
that claim could not be confirmed from the abstract, which describes GPUs and device
meshes; it has moved to §15 and nothing here depends on it.
[Graphene](https://dl.acm.org/doi/10.1145/3582016.3582018) (ASPLOS'23, NVIDIA) is the
under-cited predecessor: a tile/layout IR for tensor cores that predates linear layouts
and makes decomposition of tensors into tiles an explicit IR-level object. Hexcute's
contribution is orthogonal and portable: pick an anchor op, propagate layout
constraints outward over a DAG partitioned at memory boundaries — an algorithm that
would work with the SuperDSC `batchmatmul` as the anchor instead of `mma.sync`.

### 1.3 What these systems assume that Spyre does not provide

Almost every won abstraction bottoms out in a **thread–value map**. ThunderKittens'
16×16 tiles are defined by which lane holds which element; linear layouts index bits of
(thread, register); Hexcute's TV-layouts are literally
`(thread_id, value_index) -> coordinate`. The
[TileLang Ascend adapter](https://github.com/tile-ai/tilelang-ascend) states the
consequence plainly for NPUs — they "lack thread-level abstractions," so only
tile-granularity primitives on vector cores are offered. That port is the best available
evidence for what happens when a GPU tile DSL meets a scratchpad accelerator: the
surface language survives, the lowering is rewritten from scratch, and the
thread-mapping abstractions are simply dropped.

Four further assumptions break. A hardware scheduler with implicit placement (Triton
programs never name a core; SuperDSC names them explicitly). A shared cache giving free
cross-block reuse (on a spatial chip, reuse across cores must be an explicitly emitted
broadcast or it becomes N-fold LPDDR5 traffic). Banked shared memory with swizzling
(Spyre's constraint is 128-byte stick alignment in a per-core scratchpad — the algebra
transfers, the atoms do not). And unordered parallel grid semantics.
[Pallas on TPU](https://docs.jax.dev/en/latest/pallas/tpu/details.html) is the honest
counterexample and the closest published analog: TPUs are described as "sequential
machines with a very wide vector register," the grid executes in lexicographic order so
consecutive iterations can skip redundant HBM transfers, block shapes must have their
last two dims divisible by 8 and 128, and `dimension_semantics` is the only handle for
splitting across cores. torch-spyre's `CountedLoopSchedulerNode` is far closer to a
Pallas grid than to a Triton grid.

[TileLoom](https://arxiv.org/abs/2512.22168) (arXiv 2512.22168; the HTML v2 at
[/html/2512.22168v2](https://arxiv.org/html/2512.22168v2) carries the current title —
v1 was "TL: Automatic End-to-End Compiler…") is the paper that states this gap directly
and closes it, compiling Triton/Helion kernels onto Tenstorrent Wormhole and Blackhole
by supplying what GPU hardware gives for free. It recurs throughout this review; §6 and
§7 give its numbers. [Dato](https://arxiv.org/abs/2509.06794) states the complementary
critique — tile languages "abstract away communication details… forcing compilers to
reconstruct the intended dataflow" — and argues for streams and layouts as first-class
*types* instead, reporting 84% utilization on GEMM and 2.81× on attention against a
commercial framework on an AMD Ryzen AI NPU, and 98% of theoretical peak on Alveo FPGA.

Two more entries belong here. [Exo 2](https://arxiv.org/pdf/2411.07211) (ASPLOS 2025)
argues that custom instructions, specialized memories and accelerator configuration
state should live in *user libraries* rather than being hardcoded into compiler passes —
libraries amortized across 80+ kernels cut total scheduling code by an order of
magnitude. Spyre's OpFuncs, LX scratchpad and stick layout are exactly "custom
instructions + specialized memory + config state," so Exo 2's thesis is a direct
challenge to torch-spyre's current factoring. And the CGO 2026 Halide paper,
[Pushing Tensor Accelerators Beyond MatMul](https://arxiv.org/abs/2512.02371), uses
equality saturation for flexible tensor-instruction selection — "is this loop nest
secretly a matmul my PT unit can run?" — with one headline number (6.1× on a
downsampling routine, RTX 4070) and no non-NVIDIA demonstration.

### 1.4 The honest SOTA answer

There is a converging layout algebra, a converging execution model for GPUs, and no
consensus IR. NVIDIA's Tile IR is the loudest bid, but a third-party preprint,
[Evaluating CUDA Tile](https://arxiv.org/html/2604.23466v2), puts CuTile GEMM at
52–79% of cuBLAS on Blackwell (876 vs 1672 TFLOP/s on B200) with a 5.6×
cross-architecture gap on attention between B200 and RTX PRO 6000, while needing 22
kernel LOC for GEMM against 53 for Triton and 123 for WMMA — evidence that a tile IR
buys code size, and that performance tracks per-microarchitecture *backend maturity*.
That is precisely the risk profile of a proprietary DeepTools backend. It is also the
reason §2 matters: the alternative to authoring a tile IR is not "no IR," it is Linalg
plus a transform script, where the tiling decision stays in the compiler and becomes
data a cost model can rank. The two most transferable results in this section are older
and newer than the hype: **Roller** (§8.1) and **TileLoom** (§6.4).

---

## 2. MLIR ecosystem and the tile-IR decision

The draft cited exactly two MLIR projects, tt-mlir and CUDA Tile IR, and therefore
framed the KTIR decision as "which tile IR." That is the wrong framing. The largest
body of production MLIR work answers the question differently: keep structured loop IR,
let the compiler choose the tiling, and make the *schedule itself* an IR object that can
be printed, diffed, generated and searched. For a project whose asset is a cost model
and whose missing piece is the loop that consumes it, that answer deserves a hearing
before a new language is designed.

### 2.1 Linalg and the transform dialect: schedules as searchable data

[Composable and Modular Code Generation in MLIR](https://arxiv.org/abs/2202.03293)
(2022) is the design document for the structured-ops approach: named and generic
operations on tensors carry enough structure (iteration space, indexing maps, iterator
types) that tiling, fusion, padding, vectorization and bufferization can be implemented
once as reusable transformations rather than per-operator. Tiling is a *transformation
applied to* the IR, not a construct the user writes.

The [`transform` dialect](https://mlir.llvm.org/docs/Dialects/Transform/) closes the gap
that made this unattractive for performance work: a transformation sequence is written
in MLIR, so a schedule is a value in the compiler's own IR.
[The MLIR Transform Dialect](https://www.steuwer.info/files/publications/2025/CGO-The-MLIR-Transform-Dialect.pdf)
(CGO 2025) is the paper of record; the number that matters for adoption risk is that
expressing an existing pipeline as a transform script cost **≤2.6%** over the default
pipeline. That is the cheapest possible experiment for torch-spyre's central research
problem: the decisions the cost model would like to rank — tile factors, work division,
LX policy, restickify placement — are today spread across imperative Python passes with
no first-class representation. Making them data is a prerequisite for ranking them, and
it does not require a new language.

### 2.2 IREE: the open PyTorch → custom-accelerator path

[IREE](https://iree.dev) is the main open route from PyTorch to a non-vendor
accelerator, reached through [torch-mlir](https://github.com/llvm/torch-mlir/blob/main/docs/roadmap.md).
Three of its subsystems are directly comparable to torch-spyre components. Its own
tiling, fusion and bufferization stack occupies the layer where torch-spyre uses
Inductor plus backend passes. Its [`stream` dialect](https://iree.dev/reference/mlir-dialects/Stream/)
models execution scheduling, asynchrony and allocation explicitly — the concerns that on
Spyre are split between work division, HBM pool planning and the opaque DeepTools
schedule. And its [data tiling](https://iree.dev/community/blog/2025-08-25-data-tiling-walkthrough/)
work packs tensors into target-specific layouts early and propagates the encoding
through the graph, which is structurally the same pass as layout propagation plus
restickify insertion. IREE also exposes a documented
[tuning and codegen-configuration](https://iree.dev/reference/tuning/) interface — the
hook a cost model would drive — and the
[design roadmap](https://iree.dev/developers/design-docs/design-roadmap/) is candid
about which parts are aspirational. No performance numbers are published in the
data-tiling walkthrough or in
[iree-amd-aie](https://github.com/nod-ai/iree-amd-aie), so nothing here should be read
as a speed claim (§15).

### 2.3 Triton onto a non-GPU backend: the Linalg route

For backends that want Triton as a *frontend* without inheriting SIMT lowering, the
established path is Triton → Linalg: Microsoft's
[triton-shared](https://github.com/microsoft/triton-shared) and Cambricon's
[triton-linalg](https://github.com/Cambricon/triton-linalg) both do exactly this. This
is the practical alternative to TileLoom's approach (§6.4) of building a spatiotemporal
mapper directly. Neither publishes performance numbers, and the harder question they
sidestep is the one §1.3 raises: the parts of a Triton program that assume a thread–value
map do not survive the trip.

### 2.4 MLIR-AIE and IRON: the closest open-source analogue to Spyre

AMD's NPU stack is the closest *open* analogue to Spyre's programming model — spatial
compute tiles, explicit DMA, software-managed L1, no cache — and it is fully readable.
The [Versal AIE-ML architecture manual](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Tile-Architecture)
documents the tile; the [Linux accel driver docs](https://docs.kernel.org/accel/amdxdna/amdnpu.html)
document the host interface; the
[MLIR-AIE programming guide](https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/README.md)
documents the compiler. The importable idea is the **ObjectFifo**: a typed,
statically-sized producer/consumer channel between tiles whose depth *is* the
double-buffering decision, so buffer count, DMA descriptors and synchronization are
generated from one declaration. That is the abstraction torch-spyre currently lacks when
it needs to reason about whether a staged tile is single- or double-buffered — which is
exactly the effective-capacity question §7.1 raises and Q2 must settle.
[IRON](https://arxiv.org/abs/2504.18430) (FCCM'25) is the Python-level API above it, and
[MLIR-AIR](https://arxiv.org/pdf/2510.14871) ("From Loop Nests to Silicon", 2025) is the
higher-level lowering from loop nests to spatial placement and data movement; its venue
beyond arXiv could not be confirmed (§15). Note the two-tier shape, which recurs in §11:
IRON is the explicit escape hatch under an automatic path.

Alongside it, [Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html) — the
compiler under Pallas/TPU, cited in the draft only through user-facing docs — is the
production example of a compiler that owns tiling and layout for a scratchpad machine
while exposing a block-and-grid surface to the user.

### 2.5 What this implies for the KTIR decision

Three postures are defensible and the literature does not settle between them. Author a
tile IR (KTIR, tt-mlir's TTKernel, CuTile) and own the lowering. Adopt structured loop
IR plus searchable schedules (Linalg + `transform`, IREE, AKG) and own the search.
Or do both in two tiers (NKI beside Neuron, Gluon beside Triton, Metalium under TTNN,
IRON under MLIR-AIR). What the evidence does support is a sequencing claim: the
measured cost of making schedules first-class data is ≤2.6% compile time, while the
measured benefit of a tile IR in the one third-party evaluation available is smaller
code, not faster code. **The cheapest first step is not a new IR but
schedules-as-searchable-data** — and it is a prerequisite for every experiment in §12
regardless of which posture is eventually chosen.

---

## 3. What TorchInductor actually does — and what a PrivateUse1 backend inherits

This section is grounded in the installed PyTorch 2.11.0 source tree rather than in
summaries, because the answer to "is any of it cost-model driven" turns out to be
*almost* no — and the draft overstated the "almost" in three places, corrected below.

**A note on the baseline.** The draft asserted several PyTorch release facts — that
2.12 shipped 2026-05-19 adding `torch.accelerator.Graph`, that 2.10 deprecated
TorchScript, that 2.11 added differentiable collectives and FlashAttention-4. None was
verified in this pass and the 2.12 date sits oddly against the 2.11.0 tree actually
read, so all of it has moved to §15. What can be stated: there is no 2026 "State of
torch.compile" post; the latest inventory of limits remains
[ezyang's August 2025 post](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
(gradients delayed to the end of compiled regions, no differentiation w.r.t. returned
intermediates, no double-backward, "static by default, recompile to dynamic shapes,"
distributed collectives and DTensor compilable but "unoptimized by default," results
explicitly not bitwise-equivalent to eager), and PyTorch 2.9 added
`torch._dynamo.error_on_graph_break()` as a region-scoped alternative to the one-way
`fullgraph` flag ([2.9 blog](https://pytorch.org/blog/pytorch-2-9/)).

### 3.1 Fusion is decided in the scheduler by a byte proxy

The ranking function is `InductorChoices.score_fusion` in
[`choices.py`](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/choices.py).
It returns a four-element lexicographic tuple
`FusionScore(template_score, type_score, memory_score, proximity_score)`, where
`template_score` prefers epilogue/prologue ordering, `type_score` is the boolean
`node1.is_reduction() == node2.is_reduction()`, `proximity_score` is negated distance in
graph order, and `memory_score` is literally the summed byte size of the memory
dependencies the two nodes share. Greedy rounds run to fixpoint, capped by
`max_fusion_size = 64` and gated by `score_fusion_memory_threshold = 10`. The official
taxonomy in
["Why Is PyTorch Compile So Fast: Kernel Fusion"](https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/)
(June 2026) frames every fusion benefit in terms of memory traffic, matching what the
source does.

The empirical escape hatch, `benchmark_fusion`, compiles and times both variants and is
off by default. The draft read its guard as a proxy for "any non-Triton backend"; it is
not. `scheduler.py:4013–4015` reads `if device.type == "cpu" and config.cpu_backend !=
"triton"`, i.e. it is literally CPU-and-not-Triton. The real barrier for a PrivateUse1
backend is elsewhere and harder: `BaseScheduling.benchmark_fused_nodes` raises
`NotImplementedError`, so there is nothing to time unless the backend implements it.
Users have been asking for steerable fusion since March 2025:
[pytorch#149603](https://github.com/pytorch/pytorch/issues/149603) is still open, with
today's workarounds being no-op custom ops, graph breaks, or hand-written Triton.

### 3.2 One runtime-estimate-driven fusion decision exists — and cannot fire here

This is the draft's most consequential correction. `speedup_by_fusion` early-returns
only when `not config.benchmark_fusion and not is_multi_template`; in the
multi-template path it calls `node2._get_estimated_runtime()` (`scheduler.py:4148`) and
feeds the result to `_estimate_fused_epilogue_runtime` (`scheduler.py:2769`), which
scales the epilogue estimate by an extra-bytes ratio and compares it against measured
unfused template timings. So Inductor *does* contain one fusion decision driven by a
runtime estimate. It is gated on `MultiTemplateBuffer` plus Triton templates — machinery
a PrivateUse1 backend does not have — so it never fires for Spyre. The accurate claim is
"one such decision exists and is unreachable from here," not "there is no runtime model
in the fusion path."

The model behind it, `BaseSchedulerNode._get_estimated_runtime()` in
[`scheduler.py`](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/scheduler.py),
is a roofline — `max(flops_est / gpu_flops, counted_bytes / gpu_memory_bandwidth)` —
with `factor = 1.0` and a TODO conceding it is inadequate. Its third line is
`if not is_gpu(get_device_type(layout)): return 0`. That remains the sharpest single
finding for a non-GPU backend: **Inductor's default runtime estimate is identically zero
on Spyre.**

### 3.3 The one injection point: `estimate_op_runtime` and `device_info`

Two upstream facts the draft missed, both of which soften "inherits no timing model at
all" to "one hook exists, wired only to comms reordering."

`torch._inductor.config.estimate_op_runtime` (`config.py:442`) accepts a
**user-supplied callable** and is dispatched in
[`comms.py`](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/comms.py)
(around lines 2138–2146) when the comm/compute overlap reordering passes need a runtime
for a node. It is a real, documented extension point for a custom cost model — and it is
plumbed to exactly one consumer, which for a single-card inference backend is the one
consumer that does not matter. §14 notes the inversion: the day multi-card performance
comes into scope, the hook that is useless today becomes the natural integration point.

Separately,
[`torch/_inductor/analysis/device_info.py`](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/analysis/device_info.py)
is a datasheet table of peak TOPS and DRAM bandwidth keyed by device name. Registering
Spyre there is the minimal change that stops the roofline returning zero. It would not
make the estimate *good* — `factor = 1.0` and a two-term roofline will not model an
explicit-NoC scratchpad machine — but it converts "no signal" into "a documented, wrong
signal," which is the difference between a hook that cannot be tested and one that can.

Also relevant, and covered in §8.6 rather than here: Inductor ships its own
measure-and-persist cache (`config.runtime_estimations_mms_benchmark` plus
`get_estimate_runtime_cache()`), which is LENS's "measured anchors" pattern already
inside the framework.

### 3.4 Memory planning, tiling, and the vestigial learned models

**Memory planning: three mechanisms, different default states.** `allow_buffer_reuse`
(default on) is a same-key free list emitting `ReuseLine`s — opportunistic size
matching, not planning. The real allocator in
[`codegen/memory_planning.py`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/memory_planning.py)
is a live-range bin-packer building `TemporalSplit`/`SpatialSplit` allocation trees into
pools backed by one `torch.empty` — the closest upstream analogue to LX scratchpad
allocation, and **default off** (`TORCHINDUCTOR_MEMORY_PLANNING`). Meanwhile
[`memory.py::reorder_for_peak_memory`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/memory.py)
*is* default on and *is* genuinely cost-model-driven search: three heuristic topological
sorts (`lpmf`, `bfs`, `dfs`), each scored by an analytical peak-bytes estimator, minimum
wins. Inductor will search a schedule space when it has a cheap analytical objective —
but the objective is bytes, never time. Tiling follows the same pattern:
[`tiling_utils.py::analyze_memory_coalescing`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/tiling_utils.py)
scores candidate tilings by bytes made coalesced, weighting writes 2× reads, and takes
the argmax.

**Learned cost models exist upstream and are vestigial.**
[`torch/_inductor/autoheuristic/`](https://github.com/pytorch/pytorch/tree/main/torch/_inductor/autoheuristic)
contains offline-trained decision trees checked in as generated Python
(`_MixedMMH100.py`, `_MMRankingA100.py`, `_PadMMA100.py`) over features like
`arith_intensity`, `m*n`, `k*n` and dtype. They are hard-gated by SKU
(`shared_memory == 232448 and device_capa == (9, 0)`), and the default
`autoheuristic_use` is a single decision, `mixed_mm`. Meta built the
learned-cost-model infrastructure and it generalized to one decision on two GPU SKUs.
Read that as evidence for an analytical model over a per-SKU learned one.

### 3.5 Fusion cost models outside Inductor

The draft read as if only Inductor and vLLM existed, which made an already-answered
question look open. Four systems matter.

**XLA:GPU PriorityFusion** is the direct counterexample: a production compiler that
replaced a heuristic fusion score with an analytical latency model. Candidates are held
in a priority queue keyed by predicted runtime delta — in
[`priority_fusion.cc`](https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/priority_fusion.cc)
the priority is an `absl::Duration` computed as `current_priority + time_unfused −
time_fused` from `GpuPerformanceModel::EstimateRunTimes`, with incremental
re-prioritisation of affected producers to bound compile time. The
[RFC](https://github.com/openxla/xla/discussions/6407) and the follow-up
["Cost Models in XLA GPU"](https://github.com/openxla/xla/discussions/10065) give the
design rationale, and
[Operator Fusion in XLA: Analysis and Evaluation](https://arxiv.org/abs/2301.13062)
is the independent study. Two details are directly useful here. First, the RFC writes
the priority sign inverted relative to the shipped code; the code is authoritative
(§15). Second, and load-bearing for any γ-style overlap term,
[`gpu_performance_model_base`](https://github.com/openxla/xla/blob/main/xla/service/gpu/model/gpu_performance_model_base.h)
implements `compute + memory − min(compute, memory) · kMemoryComputeParallelism` with
`kMemoryComputeParallelism = 0.95`, while XLA's own documentation describes the model as
`max(compute, memory) + launch overhead`. A single fitted overlap scalar is an accepted
engineering point in a shipping compiler — and the doc page disagrees with the code, so
do not derive an overlap form from documentation.

**DNNFusion** ([PLDI'21](https://arxiv.org/abs/2108.13342)) classifies operators into a
small lattice of mapping types (one-to-one, one-to-many, many-to-many, reorganize,
shuffle) and uses the type pair to decide fusion *without* a cost model in the easy
cases, consulting profiling only for the ambiguous ones. That two-tier structure is
cheap to replicate and is the right shape for a backend that wants to spend model
evaluations sparingly. Its "9.3×" is a maximum against PyTorch-Mobile, not an average
(§15). **AStitch** ([ASPLOS'22](https://jamesthez.github.io/files/astitch-asplos22.pdf))
targets memory-intensive operators specifically, stitching them through a hierarchical
data-reuse scheme rather than treating fusion as a binary. **Apollo**
([MLSys'22](https://proceedings.mlsys.org/paper_files/paper/2022/file/e175e8a86d28d935be4f43719651f86d-Paper.pdf))
partitions the graph and solves fusion within partitions under explicit memory
constraints — the closest published statement of "fusion under a capacity bound" outside
the accelerator-mapping literature. **Chimera**
([HPCA'23](https://sizezheng.github.io/files/7A-3.pdf)) handles fused GEMM chains and is
worth carrying for one artifact: a closed-form optimal tile size,
`T* = −α + sqrt(α² + MC)` for capacity `C`, which is a free analytic baseline to check a
search against (used in Q3 and Q6).

### 3.6 The most serious production user, and the extension contract

vLLM maintains ten hand-written Inductor FX passes rather than trusting the scheduler
([fusions reference](https://docs.vllm.ai/en/stable/design/fusions/)): AllReduce+RMSNorm
5–20% end-to-end on Hopper/Blackwell with TP, AsyncTP GEMM+collective 7–10%,
attention+quant 3–7%, RMSNorm+quant and SiLU+Mul+quant 1–4% each, RoPE+KV-cache 2–4% on
ROCm/AITER. Their
[introductory post](https://blog.vllm.ai/2025/08/20/torch-compile.html) reports up to 8%
on Llama 3.1 405B FP8 from SiLU+quant and 15% from AllReduce+RMSNorm, and admits the
integration "uses many private torch.compile APIs and relies on unstable implementation
details." vLLM deliberately wraps attention in
`torch.ops.vllm.unified_attention_with_output` so Dynamo cannot trace in, then splits the
graph at those ops for piecewise compilation
([design docs](https://docs.vllm.ai/en/latest/design/torch_compile/)) — and confirms that
Inductor autotuning "takes seconds to minutes," which is why max-autotune ships off. The
cost of that barrier is documented from the consumer side in
[vllm#24629](https://github.com/vllm-project/vllm/issues/24629). Read as evidence: those
percentages are what the byte proxy leaves on the table when a team is willing to write
the passes by hand.

**The extension contract.** [pytorch#99419](https://github.com/pytorch/pytorch/issues/99419)
is the canonical reference: an out-of-tree backend supplies `Scheduling`, `Kernel` and
`WrapperCodegen`, registered via `register_backend_for_device`, and Inductor supplies
graph fusion and lowering. The promised "fusion for free" is exactly the part torch-spyre
switched off — and jansel's guidance in
[dev-discuss 3226](https://dev-discuss.pytorch.org/t/disabling-codegen-specific-fusions-in-torchinductor-for-per-op-kernel-generation/3226)
for one-kernel-per-op is precisely `return False` from `can_fuse()`. torch-spyre is on
the sanctioned path, not a hack. The supported installation point for a replacement
ranking policy is `config.inductor_choices_class` (`config.py:770`), which is what Q5
exercises.

---

## 4. Can a compiler discover FlashAttention?

### 4.1 Decompose the question first

FlashAttention is not one thing. In compiler terms it is four separable transformations,
and only one is genuinely hard.

**T1 — break the GEMM fusion boundary.** Stock Inductor routes matmuls to templated
Triton kernels or vendor libraries, isolating them from surrounding pointwise and
reduction ops. [Flashlight](https://arxiv.org/abs/2511.02043) names this precisely:
"this bifurcation creates a fusion boundary that isolates the GEMM from surrounding
computations." Its fix is a unified reduction IR modelling a matmul as p-dimensions
(parallel, the output m/n) and r-dimensions (reduction, the contracted k), giving every
op a *computation sketch* like `[(P0,P1),(R0)]`. The exact analogue on Spyre is the
SuperDSC matmul as an opaque atom.

**T2 — dimension demotion.** A producer's parallel dimension becomes an inner
*sequential* loop in the fused consumer. This is what manufactures the KV-block loop:
you spend parallelism to buy locality. Chained matmuls additionally need *tiling-aware
dimension elimination* — when tile size `B_Pi ≥ |P_i|` the tile-loop bound becomes 1 and
the dimension collapses out of the sketch, making otherwise incompatible sketches
fusable. That condition is exactly why small head_dim matters.

**T3 — online softmax.** The only step that is not a loop transformation: it changes the
algorithm. Stable softmax is two dependent passes and no amount of loop fusion is legal.
Flashlight's Appendix A gives the escape: if ⊕/⊗ form a ring and E is a homomorphism
(`E(a⊕b) = E(a)⊗E(b)`), the two-pass and single-pass recurrences provably converge.
`exp` qualifies because `exp(x−y) = exp(x)/exp(y)`, which is what licenses the rescale
`S_new = S_old · exp(m_old − m_new)`.

**T4 — tiling, on-chip buffer assignment, and not materializing S.** This is §7's and
§8's problem, not a fusion problem.

### 4.2 Who can do it, and with how much human input

| System | Human input | Derives online softmax? | Headline |
|---|---|---|---|
| [FlexAttention](https://arxiv.org/abs/2412.05496) | Full hand-written template | No — baked in | Only pure fns of (b,h,q,kv) expressible |
| [Flashlight](https://arxiv.org/abs/2511.02043) | None (monkey-patch on PT 2.5) | Yes, by ring/homomorphism rule | ≤1.48× over FlexAttention on score_mod |
| [Neptune](https://arxiv.org/abs/2510.08726) | Few-line schedule template | Yes, by *symbolic solving* | 1.35× geomean, 284/320 configs won |
| [Mirage](https://arxiv.org/abs/2405.05751) | None | Yes, by search | Up to 2.2× over FA/FlashDecoding |
| [Nautilus](https://arxiv.org/abs/2604.14825) | Math-level description | Yes (rolling update) | 1.22× over FA2, GH200; <1 min sched |

**Read Flashlight's numbers carefully.** Its headline 5×+ is Evoformer versus *unfused*
`torch.compile`, not versus FlexAttention — and FlexAttention cannot express Evoformer at
all. Against FlexAttention on supported variants it is up to 1.48× on `score_mod`
variants and *slower* on `block_mask` variants (FlexAttention skips fully-masked blocks;
Flashlight has no sparsity optimization). It is generally slower than FlashInfer except
on ALiBi. End-to-end AlphaFold2 (48 Evoformer layers) is 6–9% on H100/A100. See the
[MLSys 2026 artifact appendix](https://mlsys.org/virtual/2026/poster/3540) for the
per-variant breakdown. Maturity flag: the stated artifact repo,
[pytorch-flashlight](https://github.com/bozhiyou/pytorch-flashlight), presented as a bare
PyTorch fork with the stock README, 1 star and 0 forks when fetched — budget for
reimplementation, not reuse.

**Neptune is the stronger result on T3 specifically**, because it does not know about
softmax. It matches the loop structure to extract a reducer `f` and term-generator `g`,
then solves `h(t,r,r') = g(r', g_c^{-1}(r,t))` and validates distributivity, yielding
`h(t,m_old,m_new) = t·exp(m_old−m_new)` as *output*. The price is that `g` must be
invertible (Welford's algorithm is the paper's stated failure) and the user supplies a
schedule template naming `RollingUpdate`/`SplitKUpdate`.
[Nautilus](https://arxiv.org/html/2604.14825v1) reaches FA3-like kernels from a
math-like description via successive lowering (Scalar IR → VR-tile IR → MA-tile IR) with
"rolling update" online reduction fusion — but delegates actual tile codegen to
Triton/TileLang/Tawa, a layer torch-spyre would have to own itself.

### 4.3 The reality check

PyTorch has effectively conceded the compiler-generated path on Blackwell. Their
[March 2026 post](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
states FlexAttention was ~80% of FA3 at launch and is ~60% today, and that on GB200
versus cuDNN "what was once a small gap has grown to a chasm." The fix was to make
FlexAttention a frontend JIT-instantiating hand-written
[FA4](https://tridao.me/blog/2026/flash4/) via CuTeDSL — 1.6–3.2× forward and 1.85–2.3×
backward over Triton. Their stated reason: warp specialization, TMEM and deep async
ping-pong pipelines are "the kind of low-level choreography that a general-purpose
compiler can't easily discover."
[Modal's reverse-engineering write-up](https://modal.com/blog/reverse-engineer-flash-attention-4)
enumerates what that means concretely — five specialized warp roles, `exp` via a cubic
polynomial on FMA units instead of the SFU, and **conditional rescaling** that skips the
online-softmax correction unless the new max threatens numerical stability, cutting
corrections ~10×. That last one is a numerically-motivated algorithm change outside every
algebraic framework above. No compiler derives it — and on an fp16-native chip it is also
a warning that the recurrence's numerics are a first-order question, which is why Q4 now
carries an accumulate-width sub-question.

### 4.4 Why this reads differently for a dataflow chip

The GPU cautionary tale transfers less than it looks, because the choreography that
defeats compilers on Blackwell is DeepTools' job on Spyre, not torch-spyre's. Three
results matter more here than any GPU work.

[FlatAttention](https://arxiv.org/abs/2604.02110) shows that porting the FlashAttention
*dataflow* to a tile-based many-PE accelerator is the wrong target: on the same hardware,
a dataflow built on on-chip NoC collectives instead of HBM gets 4.1× speedup and 16× less
HBM traffic than the FA-3 dataflow, at 92.3% utilization and 1.9× average over GH200 on a
32×32 configuration. [Sohn et al.](https://arxiv.org/abs/2404.16629) make the point more
sharply for streaming dataflow: the winning transformation is not tiling but reordering
multiplication and division to get O(1) intermediate memory at full throughput.

And [FFM](https://arxiv.org/abs/2602.15166) (MICRO 2026) is the deepest: **always-fuse is
not optimal**, because inter-Einsum fusion competes with intra-Einsum reuse for the same
on-chip buffer, and the optimal fusion set shifts with sequence length. Pruning
Pareto-dominated partial mappings over (energy, latency, *buffer reservation*) finds
provably optimal fused mappings in 1.5 CPU-hours where SET needs ~15,000 to get within
1%, delivering 1.8× EDP over hand-optimized TransFusion.
[FuseFlow](https://arxiv.org/abs/2511.04768) independently reaches the same "full fusion
is not always optimal" conclusion for reconfigurable dataflow architectures (~2.7× on
GPT-3 with BigBird attention; a 56-point design space spanning 1.5×–3.9×), though it
requires user `Fuse{}` annotations and does not auto-detect attention. FFM is
torch-spyre's LX allocation plus tile-size search stated as a search problem with a
correctness proof — and nobody has combined Neptune-style automatic repair-term derivation
with FFM-style capacity-constrained mapping search. Note that §7.4's DeFiNES reaches the
"optimum is interior" conclusion independently and three years earlier, for fused CNN
stacks, with open tooling.

A fourth, adjacent idea worth tracking:
[Event Tensor](https://arxiv.org/abs/2604.13327) elevates tile-level completion events
into first-class multi-dimensional tensors in the IR, enabling AOT-compiled dynamic
megakernels (1.48× over vLLM at batch 1 on Qwen3-30B-A3B, 8×B200; warmup 35 s vs vLLM's
123 s). That is a plausible model for representing cross-core dependencies that
torch-spyre currently hands opaquely to DeepTools.

---

## 5. Why hand-written kernels still beat compilers

**The gap is structural, not parametric.** The 2025–26 evidence says hand kernels do not
win because humans pick better tile sizes; they win because they operate on a different
program representation, one where asynchrony, resource budgets, work-queue residency and
inter-core fabric are first-class objects.

**Resource budgets drive the schedule, not the other way around.**
[FlashMLA's "seesaw" scheduling](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md)
exists for one reason: the MLA decode output tile is 64×512, i.e. 32,768 registers per SM
— the entire register file — so there is no room for a second accumulator and FA3's
ping-pong is unavailable. DeepSeek split O vertically into two 64×256 halves and
interleaved two warpgroups over different KV block pairs, reaching 3000 GB/s and 660
TFLOPS on H800 SXM5 at ~80% tensor-core utilization. The detail that matters most for
cost-model work is the honest caveat: seesaw is ~2% *slower* than the prior ping-pong
version in memory-bound regimes. The correct schedule is a function of the operating
point. A companion write-up on
[Hopper FP8 sparse decoding](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md)
shows the critical path was a *non-matmul* conversion op (~50 cycles/token dequant vs ~34
for the MMA) and the fix used an inter-core fabric primitive: 250 → 410 TFLOPS. A
FLOP/byte model would have located neither.

**Breaking regularity assumptions.** [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
is ~300 lines, fully JIT, and reports up to 2.7× over DeepSeek's own tuned CUTLASS 3.6
implementation (1550 TFLOPS FP8 on H800). Two of its optimizations are things no compiler
would emit: *unaligned block sizes* (N = 112, chosen so ⌈7168/112⌉×2 = 128 SMs stay busy —
a wave-quantization fix, not a locality fix) and *FFMA SASS interleaving*,
post-compilation binary editing worth 10%+. Note the epilogue: NVCC 12.9 now does FFMA
interleaving automatically and DeepGEMM disabled its SASS pass. The compiler caught up on
the lowest-level trick, not the scheduling ones. Its per-block scaling machinery is not
only a kernel trick; §10 treats it as a compilation problem.

**The memory-bound case is worse than people assume.**
[QuACK](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)
hits 3.01 TB/s on H100 softmax (89.7% of the 3.35 TB/s peak) against `torch.compile`'s
1.89 TB/s (56.4%). The cause is embarrassing and instructive: Inductor emits two global
loads plus a store instead of one load plus a store, so it runs at two-thirds of
achievable bandwidth. QuACK wins with a *vanilla three-pass* softmax by spreading the
reduction across the full memory pyramid including Hopper cluster reduction. A better
algorithm lost to better placement — and the compiler's loss was pure memory-traffic
accounting, checkable statically on a loop-level IR before any hardware run.

**MoE: the win moved to algebra and fabric.**
[MegaBlocks](https://arxiv.org/abs/2211.15841) (MLSys'23) reformulated MoE as
block-sparse ops to remove the drop-vs-pad tradeoff (40% over Tutel, 2.4× over
Megatron-LM). Triton reimplementations converged: PyTorch's
[persistent cache-aware grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
sets grid = 132 (H100 SM count), adds grouped launch ordering (+60% L2 hit rate, 1.33×),
and builds *device-side* TMA descriptors because the expert index is runtime data —
1.42–2.62× end-to-end on 16B DeepSeek-v3.
[TritonMoE](https://arxiv.org/html/2605.23911v1) runs on A100 and MI300X unchanged and
beats Megablocks at 32–128 tokens, but is honest that hand-tuned CUDA still wins at
2048+; it also reports permute/unpermute at <3% of runtime while expert FFN is >95%,
which is a warning that the dispatch overhead everyone optimizes is not where MoE time
goes. [SonicMoE](https://arxiv.org/abs/2512.14080) (ICLR 2026) changes the math:
fine-grained MoE arithmetic intensity is O(min(d/G, Tρ)) — ~210 for Qwen3-Next against
~2570 for a dense MLP — so it fuses gather into the GEMM prologue (X never materialized),
fuses scatter with aggregation, and reassociates the backward contraction to avoid
caching Y and dY: 45% activation memory cut, 1.86× over ScatterMoE. Its
[Blackwell blog](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/) reports different
figures (see §15). And Cursor's
[Mixture-of-Kittens](https://cursor.com/blog/mixture-of-kittens) fuses all MoE
communication and compute into one deterministic megakernel on GB300 NVL72, statically
partitioning SMs into "comp" and "comms" roles with pull-based dispatch (+29% NVLink
utilization; signaling 103 µs → 18 µs): 2.37× layer-level, 1.41× end-to-end over 512
GPUs. Conceptually that is a work-division decision applied to communication — and it is
retained here deliberately as an orphan, since multi-card performance is out of scope
(§14).

**Even hand kernels miss the ceiling.**
[ThunderKittens](https://arxiv.org/abs/2410.20399) (ICLR 2025) opens by asserting that
"hand-written custom kernels fail to meet their theoretical performance thresholds, even
on well-established operations like linear attention," and backs it with 14× over Flash
Linear Attention, 6.5× on learned feature maps, >3× on Mamba-2, 10–40% over FA3 on
attention backward, and cuBLAS-competitive GEMM (855 TFLOPs, 86% of peak) in under 100
lines. [HipKittens](https://arxiv.org/abs/2511.08083) extends the argument: peak AMD
kernels are written in raw assembly, and tile abstractions port but the algorithms
instantiating them must be rethought per architecture. That is the closest published
statement of torch-spyre's own position: portable abstraction, non-portable schedule.
Dispatch overhead is quantified by HazyResearch's
[megakernel post](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) — ~5 µs
of stall per kernel boundary, cutting H100 Llama-1B from a theoretical 1350 to ~770
forward passes/s; their fix is an explicit 16 KiB shared-memory page allocator plus
global counters, reaching 78% of H100 memory bandwidth against ≤50% for vLLM/SGLang.

**The counter-movement is exactly the torch-spyre research area.**
[Twill](https://arxiv.org/abs/2512.18134) formulates joint software pipelining plus warp
specialization as ILP (modulo scheduling for minimum initiation interval) plus SMT
(memory capacity, register budgets, warp assignment) and rediscovers the FA3 schedule to
within 1% (~645 TFLOPS, H100, seq 16384) and FA4 to within 2% on B200 — noting pointedly
that a year elapsed between Hopper's release and FA3. The
[Triton warp-specialization roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/)
describes the same machinery as six passes (data partitioning, SWP loop scheduler,
partition scheduler, SMEM/TMEM buffer creation, memory planner with channel-aware
liveness, code partitioner), reaches within 10–20% of cuDNN on B200 flash-attention
forward, and names its future direction "model-based global optimization": static or
profiled cost models pruning the joint space of partitioning, synchronization,
scheduling, memory planning and layouts. Meanwhile
[Gluon exists](https://www.lei.chat/posts/gluon-explicit-performance/) because Triton's
automatic path could not express layouts, shared-memory allocation and warp
specialization, with the frank admission that "to carefully arrange all instruction
issuing statically with a compiler is a very hard problem." For torch-spyre all of this
is *descriptive* rather than prescriptive — the knobs belong to DeepTools — but it is the
clearest statement anywhere of what a cost-model-driven joint search is for.

**LLM kernel generation is real, scored against weak baselines, and NVIDIA-skewed.**
[KernelBench](https://arxiv.org/abs/2502.10517) (250 tasks, L40S) reports one-shot fast₁
of 12%/36%/2% for DeepSeek-R1 across levels, with functional correctness — not speed — as
the persistent failure, and notes CUDA is 0.073% of The Stack. Stanford CRFM's
[search-over-hypotheses run](https://crfm.stanford.edu/2025/05/28/fast-kernels.html) got
179.9% of torch on conv2d and 484.4% on LayerNorm — but 52% on FP16 matmul and 9% on FP16
FlashAttention. The moment a real hand kernel is the baseline, the gap reopens.
[CUDA-L2](https://arxiv.org/abs/2512.02551) claims +19.2% over cuBLAS on HGEMM with a
1,000-configuration budget. A [2026 survey](https://arxiv.org/html/2601.15727v3) reports
fast₁ up to 70% on KernelBench L1 while flagging reward hacking and near-zero coverage of
non-NVIDIA hardware. That last point applies to this review too, and the correctives —
AMD's GEAK, [KernelGenBench](https://arxiv.org/abs/2607.27231) (multi-source, multi-chip),
[BackendBench](https://github.com/meta-pytorch/BackendBench) (which evaluates *whole
backends*: 271 ops for correctness and 124 for performance against OpInfo and HF-traced
shapes, and is the obvious correctness harness for a PrivateUse1 backend), plus Sakana's
CUDA Engineer as the canonical reward-hacking case study and Kevin-32B — are named on the
critique's authority and were not independently verified here (§15).

The transferable idea in this line is not the numbers but the guard:
[AutoMegaKernel](https://arxiv.org/abs/2606.09682) uses a frozen schedule-IR validator
that statically certifies deadlock- and race-freedom (zero false-accepts across 6,091
unsafe schedules) — a Spyre analogue would let an automated search over work division and
LX plans reject unsafe candidates before touching scarce hardware.
[KForge](https://arxiv.org/abs/2511.13274) is the closest attempt at synthesis for
backends the model was not trained on, using profiling feedback rather than hand-coded
heuristics.

Two closing entries.
[Modular's structured Mojo kernels](https://www.modular.com/blog/structured-mojo-kernels-part-1-peak-performance-half-the-code)
are the counterpoint to "abstraction costs performance": decomposing into
TileIO/TilePipeline/TileOp cut an SM100 matmul from 14,683 to 7,634 lines at equal ~1770
TFLOPS, because compile-time metaprogramming leaves no runtime trace. And Tenstorrent's
[TT-Metalium guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
is the closest public analogue to torch-spyre's situation: every op is hand-written as
reader/compute/writer kernels over software-managed circular buffers, with
TTNN/tt-MLIR/tt-Forge layered on top. On a dataflow accelerator, the producer/consumer
split that GPUs call warp specialization is the *default* programming model — and
Tenstorrent's compilers still defer to humans at exactly that layer. (HazyResearch's
[Retire the Abstractions](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
argues agents make DSLs unnecessary; it contains no measurements and should be read as
opinion.)

---

## 6. Dataflow-accelerator compilers, and IBM's own precedent

### 6.1 The structural break

A GPU compiler optimizes *against* a machine already trying to hide latency: caches,
MSHRs, warp schedulers, speculative fetch. Its job is largely to keep occupancy high and
avoid pathologies. A dataflow compiler has no such partner. On Spyre, Tenstorrent,
Trainium, MTIA, Groq, Cerebras and the IPU there is no cache hierarchy at all. The
crispest public statement is Tenstorrent's, via a
[third-party architectural deep dive](https://blog.gpu.net/posts/2026/june/new-blog-june12/):
"Data lives in DRAM, in another core's SRAM, or in this core's SRAM — software moves it
explicitly via DMA." Four consequences follow, and each has its own home in this review:
allocation moves into the compiler and becomes co-dependent with tiling (§7); layout
compatibility becomes a correctness problem (§6.2, §6.3); placement and routing become
compile-time combinatorial problems (§6.4); and cost models become simultaneously more
viable and harder to validate (§6.5).

### 6.2 IBM's own precedent: onnx-mlir NNPA, and the origin of the word "stick"

The draft omitted this entirely, which was its largest structural gap. IBM already ships
a production MLIR compiler for a stick-layout accelerator, in the open, with a
measurement-fitted analytical cost model inside it.

**The hardware contract.** zDNN is the runtime library for the IBM Z Integrated
Accelerator for AI. Its
[`zdnn_private.h`](https://github.com/IBM/zDNN/blob/main/zdnn/zdnn_private.h) defines
`AIU_BYTES_PER_STICK` = 128 and, above it, a page of 32 sticks (4 KiB); the
[README](https://github.com/IBM/zDNN/blob/main/README.md) documents zTensors and the
`transform` API that converts a normal tensor into stickified form and back. The term
torch-spyre uses for a 128-byte aligned chunk comes from here. Two granularities, not
one, is a fact of the published model — a point §11.2 returns to, since the torch-spyre
model currently carries only the stick.

**The compiler.** [onnx-mlir](https://github.com/onnx/onnx-mlir)
([design paper, 2020](https://arxiv.org/abs/2008.08272); shipped to users as
[zDLC](https://github.com/IBM/zDLC)) lowers ONNX through two accelerator dialects.
**ZHigh** is operation-level and layout-aware, with explicit `Stick` and `Unstick`
operations; **ZLow** is memref-level, where the stick layout is expressed as a plain
affine map on a 4 KiB-aligned memref before calls into zDNN.
[AddCustomAccelerators.md](https://github.com/onnx/onnx-mlir/blob/main/docs/AddCustomAccelerators.md)
documents the extension contract — the direct counterpart of pytorch#99419 — and
[the NNPA how-to](https://onnx.ai/onnx-mlir/AccelNNPAHowToUseAndTest.html) documents the
user-facing flags.

**The layout-conversion passes are the published ancestor of `insert_restickify` /
`optimize_restickify`.** Three are worth copying outright. First, decompositions in
[`RewriteONNXForZHigh.cpp`](https://github.com/onnx/onnx-mlir/blob/main/src/Accelerators/NNPA/Conversion/ONNXToZHigh/RewriteONNXForZHigh.cpp)
fire only after an explicit *conversion-count* benefit test — the pass asks whether the
rewrite reduces the number of stick/unstick operations before applying it. That file also
carries `SplitLargeMatMul`, which splits matmuls exceeding NNPA's dimension limits, i.e.
work division decided in the compiler for a hardware constraint. Second, the
[ZHigh transforms](https://github.com/onnx/onnx-mlir/tree/main/src/Accelerators/NNPA/Transform/ZHigh)
include `FusionOpStickUnstick`, which folds the conversion into the consuming compute
operation rather than emitting it as a separate pass over memory. Third,
`ZHighConstPropagation` stickifies constant weights at compile time, so weights are never
stickified at run time at all — the single cheapest layout optimisation available and one
that has no analogue in a torch-spyre pass list.

**Layout conversion is priced inside a placement decision.**
[`DevicePlacementHeuristic.cpp`](https://github.com/onnx/onnx-mlir/blob/main/src/Accelerators/NNPA/Conversion/ONNXToZHigh/DevicePlacementHeuristic.cpp)
runs a five-round fixed point over CPU-versus-NNPA assignment in which stick/unstick cost
is charged **per graph edge**. This is the concrete thing torch-spyre's model does not yet
do: layout conversion appears in its reports but not inside any placement or fusion
objective.

**And there is already an analytical, measurement-fitted cost model for stick
hardware.** [`PerfModelArch15.inc`](https://github.com/onnx/onnx-mlir/blob/main/src/Accelerators/NNPA/Conversion/ONNXToZHigh/PerfModelArch15.inc)
(with an Arch14 sibling), generated by
[`utils/NNPAOpPerfModel`](https://github.com/onnx/onnx-mlir/tree/main/utils/NNPAOpPerfModel),
holds per-operation regressions fitted against measured z16/z17 hardware with the
coefficient of determination recorded in the source. The matmul term blends `ceil(·,64)`
element-level work with `ceil(·,32)` page-level work — the two-granularity structure
again. Three r² values are worth memorising because of what they are:

| NNPA operation | r² in `PerfModelArch15.inc` |
|---|---|
| Unstick | 0.429 |
| Stick | 0.691 |
| MatMul_3ds | 0.706 |

The worst-fitting terms in IBM's own shipped model are the layout conversions. Read that
twice. It is a caution — layout conversion is empirically the hardest thing to model on
this hardware family, so torch-spyre's restickify outliers are not anomalous — and it is
also a reassurance, because a model with r² = 0.43 on its worst op is still good enough to
drive a production placement pass. What it does *not* provide is any rank-quality metric:
the residuals bound the model's *fit*, not its *decision quality*, which is precisely the
gap Q1 targets.

**Two engineering practices to steal.** `NNPACompilerOptions.cpp` exposes per-pass
disable flags (`--nnpa-disable-*`, `--disable-zhigh-*`), which is a ready-made ablation
harness for Q2. And the test suite asserts `TEST_INSTRUCTION` — *which* accelerator
primitive the pass chain emitted — not merely that the numerics match. A compiler whose
correctness tests only check numbers cannot detect a silent regression from a fused
conversion to an unfused one.

**Compile time is measured, not assumed.** The
[Asia LLVM 2025 talk](https://llvm.org/devmtg/2025-06/slides/technical-talk/le-onnx.pdf)
reports BERT-base compiling in 174.10 s — 18.6% MLIR, 48.3% LLVM opt, 31.4% llc — reduced
to 139.01 s by exporting constants to an external file before invoking the LLVM tools.
That is the local calibration for the roadmap's "a few minutes per priority model" budget
and it feeds Q9.

**The limits of the analogy.** NNPA invokes one zDNN primitive at a time. There is no
multi-core work division, no scratchpad allocation, and no compute/transfer overlap term.
So onnx-mlir is the published ancestor of the *layout* half of torch-spyre's problem and
of its cost model's structure — not of its tiling, allocation or work-division half. That
split is exactly what §11.3 uses to state what remains novel.

### 6.3 DNNDaSher: layout compatibility as a correctness constraint

[DNNDaSher](https://ieeexplore.ieee.org/document/10596296/) (Sen, Jain, Krithivasan,
Venkataramani, Srinivasan, *IEEE Micro* 2024, DOI 10.1109/MM.2024.3423750) is the AIU-side
statement of the same problem and the most Spyre-relevant paper in this review. On AIU
"functional correctness hinges on maintaining dataflow compatibility between
producer–consumer operations"; the framework inserts shuffle operations to reconcile
mismatches, then eliminates and coarsens them, reporting **1.27×–4.12× (avg 2.3×)**
end-to-end latency improvement on four CNN/Transformer benchmarks in measured AIU cycles.
The `insert_restickify` / `optimize_restickify` path addresses the same problem; 2.3× is
the published size of the prize, and it is a compiler-side prize. Access caveat in §15:
the IEEE page is paywalled and the numbers come from IBM Research's own publication page.

### 6.4 Placement, routing, and four families

The industry splits into four families, and only the fourth is Spyre's:

| Family | Examples | Compiler owns | Nearest lesson for Spyre |
|---|---|---|---|
| Fully spatial | SambaNova RDU, Cerebras WSE | Whole-graph placement + routing | Fusion is free; 2×–13× vs unfused |
| Cycle-deterministic static | Groq TSP | Every cycle, in time and space | Cost model *is* the schedule |
| BSP | Graphcore IPU | Exchange routes, buffer sizes | Whole-graph liveness allocation |
| Tiled multicore, SW-managed L1 + NoC | Tenstorrent, MTIA, Trainium, **Spyre** | Tiling, allocation, core mapping | All of §6–§8 |

Tenstorrent's [TTNN Optimizer spec](https://docs.tenstorrent.com/tt-mlir/specs/ttnn-optimizer.html)
is the closest published analogue to an LX planner: `LegalTensorLayoutAnalysis` enumerates
L1-vs-DRAM × interleaved-vs-sharded × tiled-vs-row-major × grid shape,
`LegalOpConfigAnalysis` expands with per-op params, `GreedyMemoryLayoutPropagation`
beam-searches at width 8, and `GreedyL1SpillManagement` evicts by farthest-next-use. Two
findings there should change how LX is thought about: they abandoned an exact
`ShardSolver` because "complex backtracking rarely provided practical benefit," and they
observed that in real models "spills are constraint-driven, not memory-driven" — 40–94% L1
headroom remained even when spills fired (this is Q2). Their planned "Cost Mode," which
would replace the heuristic score with `getOpRuntime()` estimates, is unimplemented. Their
[dialect stack](https://docs.tenstorrent.com/tt-mlir/overview.html) —
TTIR → TTNN → D2M → TTKernel → TTMetal — is a useful reference for judging whether
SuperDSC carries too many concerns at once.

[TileLoom](https://arxiv.org/abs/2512.22168) is the best evidence that a *compiler* can
win on this class of hardware. It compiles Triton/Helion tile kernels onto Wormhole (8×8
mesh, 108 MB SRAM) and Blackhole (12×10, 180 MB) via spatiotemporal mapping,
spatial-reuse-driven NoC broadcasts, temporal load hoisting, lifetime-analysis-based
buffer pruning, and a hierarchical performance model at 17% geomean error:

| Kernel | TileLoom vs TTNN (Wormhole / Blackhole) |
|---|---|
| FlashAttention | 1.94× / 1.98× |
| Mamba chunk scan | 27.23× / 16.27× |
| GEMM | 0.95× / 1.10× |
| Flash decode | 0.84× / 0.87× |

That asymmetry is the honest answer to "can a compiler discover FlashAttention on this
class of hardware": yes for *fusion-shaped* wins, no for hand-tuned steady-state kernels.
Crucially, 17% absolute error sufficed because the model *ranked* correctly, and top-2
profiling over the ranking added only +4.7%.

The rest of the field, briefly. [SN40L](https://arxiv.org/abs/2405.07518) (MICRO 2024)
reports 2×–13× over an unfused baseline, 3.7× over DGX H100 and 6.6× over DGX A100 on 8
RDU sockets with a three-tier SRAM/HBM/DDR system; SambaNova's
[framing post](https://sambanova.ai/blog/why-dataflow-matters-more-than-ever) is thin on
extractable numbers, so cite the paper. Cerebras
[solves placement and routing across ~900k PEs at compile time](https://www.cerebras.ai/blog/supporting-pytorch-on-the-cerebras-wafer-scale-engine)
and — a pragmatic pattern worth copying — matches against a hand-written kernel library
*first*, auto-generating only the residue via polyhedral techniques; their
[CSL SDK](https://sdk.cerebras.ai/) is the far end of the alternative, exposing PE-level
placement and routing to the user. Groq's
[TSP](https://dl.acm.org/doi/10.1145/3470496.3527405) (ISCA 2022) does 2-D scheduling of
instructions and data in time and space with cycle-accurate knowledge and no caches (see
also this [independent write-up](https://blog.codingconfessions.com/p/groq-lpu-design);
Groq's own [marketing post](https://groq.com/blog/the-groq-lpu-explained) should not be
cited for its 10× SRAM-bandwidth claim). Graphcore's
[Poplar](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/map-model-to-ipu-system.html)
allocates data memory at compile time by reasoning about liveness across the whole graph —
the canonical prior for LX planning, solved for a machine where the model fits on chip.

MTIA is the nearest strategic peer:
[MTIA v2](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/)
is an 8×8 PE grid, 384 KB local per PE, 256 MB on-chip SRAM at 2.7 TB/s, 128 GB LPDDR5 at
204.8 GB/s, driven through PyTorch 2 + TorchInductor with a Triton-MTIA backend
(peer-reviewed companion: [ISCA 2025](https://dl.acm.org/doi/10.1145/3695053.3731409),
which could not be extracted). *My arithmetic, not a sourced claim:* MTIA v2 and Spyre
have essentially identical off-chip bandwidth (204.8 vs 204 GB/s) but MTIA carries 256 MB
of SRAM against Spyre's 32 × 2 MiB = 64 MiB, so Spyre is ~4× more bandwidth-constrained
per byte of scratchpad — an argument for weighting working-set-reduction and LX-residency
passes heavily. Treat with caution: Spyre's two-level scratchpad may make the comparison
unfair. Trainium is the best-documented example of a hard architectural tiling constraint:
NeuronCore-v3's SBUF is
[28 MiB as 128 partitions × 224 KiB with the partition dim fixed at 128](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html),
the direct analogue of the 64-element stick — and AWS shipped
[NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html), a
bare-metal tile DSL, *beside* the automatic compiler, which is the escape-hatch question
torch-spyre faces for KTIR.

### 6.5 Cost models are more viable here, and much harder to validate

Determinism is a gift — absent caches and dynamic warp scheduling an analytical model can
in principle be exact. But when the overlap is decided inside an opaque backend,
first-principles modelling fails badly. [LENS](https://arxiv.org/html/2606.18042v2) (2026)
reports that applying GPU latency-modelling methodology to NPUs (Inferentia2, TPU
v4/v5e/v6e) gave up to **493% error**, because "the compiler can schedule adjacent
operations to execute concurrently on separate engines," and that prior simulators hit
232–323% error on element-wise ops; static shape bucketing also makes latency a step
function, defeating interpolation — a hint the draft failed to follow and §9 now does.
Their fix — two measured end-to-end anchors per bucket, composed analytically — reaches
2.15% mean error. The lesson for a γ-style overlap term is that measured anchors beat
derivation when the scheduler is opaque, and XLA's shipped
`kMemoryComputeParallelism = 0.95` (§3.5) is the precedent that one fitted overlap scalar
is an accepted engineering point.

Relatedly, an [ICPE 2024 evaluation](https://arxiv.org/html/2311.04417v3) of IPU/RDU/GPUs
found SambaFlow throughput "becomes unstable when matrix sizes grow to 2500 and above, due
to unstable PCUs and PMUs mapping through the SambaFlow compiler" — dataflow compilers
exhibit shape-dependent performance cliffs that GPUs do not, which is exactly the failure
mode a cost model must predict and the reason per-config verification (not just RMS error)
is mandatory. Finally, calibrate expectations:
[AccelOpt](https://arxiv.org/abs/2511.15915) moved Trainium kernels from 49% → 61% of peak
(Trn1) and 45% → 59% (Trn2) with aggressive agentic optimization. Even optimized dataflow
kernels live near 60% of peak. Meta's
[KernelEvolve](https://arxiv.org/abs/2512.23236) shows that even a hyperscaler with its own
Inductor/Triton backend still runs a separate kernel-authoring pipeline for its dataflow
chip (100% correctness on 250 KernelBench problems and 160 ATen ops across three
platforms; no speedup figures in the abstract).

### 6.6 The Spyre-side sources, and the production consumer

For completeness on the IBM side: the ancestral
[DeepTools](https://research.ibm.com/publications/deeptools-compiler-and-execution-runtime-extensions-for-rapid-ai-accelerator)
paper (DeepRT + RaPiDLib, same authors as DNNDaSher — the lineage claim is an inference,
§15), the
[Spyre hardware disclosure](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator)
(32 active cores on a bidirectional ring, 2 corelets each with an 8×8 SIMD-systolic array
plus two 1D fp32 vector arrays, 16 LPDDR5 channels at 204 GB/s, 5 nm, 25.6 B transistors),
the [PyTorch enablement post](https://research.ibm.com/blog/pytorch-support-ibm-spyre),
the [1H-2026 roadmap](https://dev-discuss.pytorch.org/t/ibm-spyre-accelerator-pytorch-enabling-status-and-feature-plan-1h-2026/3319)
(inference-only FP16/FP8, 7 priority models, "a few minutes per priority model" compile
budget, multi-card performance out of scope), the
[KTIR frontend](https://github.com/torch-spyre/ktir-mlir-frontend), and
[IBM/deepview](https://github.com/IBM/deepview) — whose layer-wise divergence mode is the
natural place to hang measured-versus-predicted per-op runtime.

One consumer is missing from that list and it is the one that defines the objective
function. [vllm-spyre](https://github.com/vllm-project/vllm-spyre), now
`torch-spyre/sendnn-inference`, compiles a prefill graph and a decode graph per shape
bucket and rejects requests outside the compiled set. The cost model's ultimate output is
therefore a *bucket set under a compile budget*, not a single loop nest. §9 develops this
in full; it is flagged here so §6 is not read as a complete account of the stack.

---

## 7. On-chip memory, tiling, and joint optimisation

The draft's central claim in this area — that nobody co-optimises tile shape and
scratchpad allocation — was wrong, and correcting it makes the surviving gap both smaller
and much better defined. This section is organised as four literatures that all touch
capacity, followed by the one thing none of them does.

### 7.1 Buffets, and what "capacity" actually means

[Buffets](https://ysshao.github.io/assets/papers/Buffet_ASPLOS19_Final.pdf) (ASPLOS 2019,
pp. 137–151 — the draft mis-cited the venue as ISCA) is the taxonomy the rest of this
section needs: it separates *explicit decoupled data orchestration* from caches and from
plain scratchpads, and gives the storage idiom a name and an implementation. Its measured
claims are 2% control overhead over an 8 KB RAM and 1.53× / 5.39× EDP improvement over
double-buffered DMA and over caches respectively; figures like "3–5%" and "2–3×" that
circulate in web summaries are wrong (§15).

The practical consequence for torch-spyre is a definition question that has to be settled
before any capacity-driven experiment means anything. In an explicit-orchestration design
with double buffering, the tiler's effective budget is **half** the physical scratchpad,
because a fill for iteration *i+1* overlaps the drain of iteration *i*. AKG encodes
exactly this rule as "≤ half the buffer capacity" (§7.5), and Pallas makes it a user
parameter, defaulting to
[2 buffers per input/output](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
with `pl.Buffered(buffer_count=...)` to override. Whether DeepTools double-buffers LX
tiles is unverified (§15); if it does, the LX budget in every planner and every sweep in
this project is 1 MiB/core, not 2 MiB. That is the first thing Q2 should establish, and it
changes the answer to Q6.

Production allocators are plainer than the literature. The
[tt-metal allocator](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/memory/allocator.md)
is a first-fit free list over banks, DRAM bottom-up and L1 top-down to avoid colliding
with circular buffers, with L1-versus-DRAM residency chosen *by the programmer*.

### 7.2 The packing line: allocation as feasibility

[TelaMalloc](https://dl.acm.org/doi/10.1145/3567955.3567961) (ASPLOS'23, Google)
formalizes exactly the LX problem — static control-flow graph, buffers with known
start/end times and sizes, hard capacity bound, NP-hard — and interleaves a
constraint-programming solver with a heuristic search that picks the next block to place
(longest lifetime, then largest size, then largest area), with phase decomposition and
"smart backtracking." Its headline number is *compile time*: up to two orders of magnitude
faster than the tuned production ILP it replaced, shipping in Pixel 6 and TPUv4, and
enabling models that previously could not be compiled at all.
[MiniMalloc](https://research.google/pubs/minimalloc-a-lightweight-memory-allocator-for-hardware-accelerated-machine-learning/)
attacks the identical problem with recursive DFS over a lattice of canonical solutions
plus spatial inference and dominance pruning, claiming orders-of-magnitude further
speedup. Its open-source input format is literally `id, lower, upper, size` — and that
format is the tell: **there is no cost model anywhere in this line of work.** Allocation
succeeds or fails, and failure is punted upstream.

[COSMA](https://arxiv.org/abs/2311.18246) is the cleanest spill formulation in this line —
an ILP with per-timestep create/preserve/spill/retrieve binaries plus address variables —
reporting 84% average reduction in non-compulsory off-chip accesses versus TFLite-allocator
baselines, solving nine of ten hand-designed DNNs in under 1 s average. But its objective
is bytes, and it explicitly does not model operator tiling.

### 7.3 The mapping line: capacity as a validity predicate

Timeloop's mapper enumerates a mapspace and rejects any mapping whose tile exceeds a
buffer level ("mapped tile size 62153 exceeds buffer capacity 32768" is a literal
user-facing error); the analytical model only ranks survivors
([Timeloop](https://accelergy.mit.edu/timeloop.pdf)).
[CoSA](https://arxiv.org/abs/2105.01898) (ISCA'21) makes this elegant by moving the
constraint *inside* a MIP — utilization log-linearized so the sum of log-prime-factors
assigned to a level stays under log(capacity) — with a weighted proxy objective
(−w_U·Utilization + w_C·Compute + w_T·Traffic, weights micro-benchmarked), reporting 1.5×
geomean over Timeloop's hybrid search (2.5× under an NoC simulator), 5.2× over random, and
90× faster time-to-solution (4.2 s vs 379.9 s per layer). Its motivating data is the LX
problem verbatim: 40K valid schedules for one ResNet-50 layer, 7.2× best-to-worst spread,
and **half of randomly sampled tilings violate buffer capacity**.
[Marvel](https://arxiv.org/abs/2002.07752) reduces the same search by optimizing the
off-chip subspace first (off-chip movement being 2–3 orders of magnitude more costly), a
decomposition that maps onto LPDDR5→LX staging versus within-LX reuse.
[Interstellar](https://arxiv.org/abs/1809.04070) is the load-bearing negative result: with
proper loop blocking most dataflows land within noise of each other, and *memory hierarchy
sizing* is what matters (1.8×–4.2× energy). For a fixed 2 MiB/core LX, that says spend
effort on tile sizing and residency, not on enumerating dataflow variants. TileLoom (§6.4)
confirms the pattern from the closest possible architecture: when it considers hoisting a
load outward for temporal reuse it computes the required footprint and *discards* options
exceeding capacity, before the performance model runs — and never asks whether a discarded
option would have won with a little spilling.

### 7.4 Joint tiling and residency: the claim the draft got wrong

**[Welder](https://www.usenix.org/system/files/osdi23-shi.pdf) (OSDI'23) is the single
largest omission from the draft.** It represents the whole model as a *tile-graph* in
which each node carries a tile shape and each edge a memory level at which producer and
consumer connect, and schedules data movement across the hierarchy holistically with a
tile-traffic cost model — explicitly trading intra-operator reuse against inter-operator
reuse rather than treating fusion as a separate binary decision. Capacity is *inside* the
search: candidate tile configurations are checked with `MemFootprint` against the level's
`capacity` and rejected above it. Its compile-time figures are the ones to carry: BERT in
**244 s over 651 trials** against Ansor's **15,285 s over 8,000**. Two structural details
matter downstream. Welder profiles at the connecting level rather than deriving everything
(`d.Profile`), which is LENS's measured-anchor pattern in a tile compiler. And its
tractability rests on an **inter-layer independence** argument that lets it decide one
layer's tile configuration given the connecting level; whether that survives a
per-core-partitioned scratchpad with unequal tail cores is unverified and is the core of
Q3.

[Ladder](https://www.usenix.org/system/files/osdi24-wang-lei.pdf) (OSDI'24) extends the
same lineage to data *type* and layout: a `tType` carries element width, element shape and
conversion functions, and is transformed as tiles move across memory layers, with
alignment hints (`GetDeviceHint`) playing the role Roller's alignment quantum plays for
shape. It reports roughly 2.3× average over vLLM, and the companion BitBLAS covers W4A16,
W2A16, W1A16, NF4 and FP4/FP8 variants. This is the only work treating low-bit formats as
a *general compilation* problem rather than a kernel trick, which is why §10 leans on it —
and its primary PDF returned 403, so its numbers are secondary-sourced (§15).

[Rammer](https://www.usenix.org/system/files/osdi20-ma.pdf) (OSDI'20) completes the trio:
inter- and intra-operator scheduling unified at compile time by expressing operators as
`rTask` collections mapped onto `vDevice` execution units. That is the published prior art
for `work_division.py` and `core_mapping.py` and the draft cited nothing for them.

**The KU Leuven layer-fusion line does the capacity-constrained fused version, with open
tooling, and predates FFM by three years.**
[ZigZag](https://arxiv.org/abs/2007.11360) is the underlying mapping DSE, validated within
5–7.5% against Eyeriss and ENVISION, and reports a 4.7× energy span across temporal
schedules for a single layer. [DeFiNES](https://arxiv.org/abs/2212.05344) (HPCA'23) is the
one to read for Q6: it sweeps tile size × overlap-storing mode × fuse depth with analytical
memory modelling, validated within 3% against DepFiN, and produces exactly the
capacity-versus-traffic curves this project needs for *fused* stacks. Its two findings are
directly load-bearing — the optimum is **always interior** (neither fully layer-by-layer nor
fully depth-first wins), and the best top-level memory assignment flips per tile type as
the working set crosses each level. It also measures a 26× energy span across tile size and
mode, and it costs about 18 h for a 108-point sweep.
[Stream](https://arxiv.org/abs/2212.10612) generalises to multi-core layer-fused
scheduling and is the closest formulation to LX: per-core capacity `C_j` appears **directly
as a constraint in the ILP** alongside the tiling variables, and it is validated at 96–97%
against three real chips.

Three more belong here. [TileFlow](https://gulang2019.github.io/files/tileflow-micro23.pdf)
(MICRO'23) does fusion-mapping DSE for fused dataflow accelerators, models 384 KB per core
across four cores, validates within 5.4% against RTL over 131 hand-written mappings, and —
relevant to Q4 — matches FLAT at an order of magnitude lower L1 footprint by tiling the
softmax column dimension that FLAT leaves whole. [T10](https://arxiv.org/abs/2408.04808)
(SOSP'24) co-optimises the *partition* across a mesh of cores, which is the variable Spyre
fixes upstream (abstract-only, §15). And [Elk](https://arxiv.org/abs/2507.11506) (MICRO'25)
comes closest to pricing capacity: it partitions each core's SRAM between execution space
and preload space, i.e. it treats the split of a fixed budget as a decision variable rather
than a constraint. It does not fuse.

**So the correct statement is:** joint tile-shape and on-chip-residency optimisation is
well established, on GPUs (Welder, Ladder), on NPUs (AKG), and on multi-core dataflow
accelerators in simulation (Stream, DeFiNES, TileFlow, Elk). What is unclaimed is stated
in §7.6.

### 7.5 Polyhedral memory promotion: the oldest joint formulation

The draft omitted polyhedral compilation entirely, which is awkward because the polyhedral
line has been solving "choose tile sizes and promote tiles into a software-managed
scratchpad, jointly, as one affine problem" since 2008.

[Baskaran et al.](https://www.ece.lsu.edu/jxr/Publications-pdf/tr5-08.pdf) (PPoPP 2008) is
the original: automatic data movement into explicitly-managed local memories, with tile
sizes and local-memory footprint expressed as one constrained optimisation over the affine
representation. PPCG carries the production version — memory promotion into shared and
private memory, bounded by `--max-shared-memory`, where a promotion that does not fit is
simply *refused* ([`ppcg_options.c`](https://raw.githubusercontent.com/Meinersbur/ppcg/master/ppcg_options.c)).
MLIR ships the same idea in
[`AffineDataCopyGeneration`](https://mlir.llvm.org/doxygen/AffineDataCopyGeneration_8cpp_source.html):
given a `fastMemoryCapacity`, it generates copies for a loop, and when the copy does not
fit it *recurses to the next inner loop*. Note the shape of both answers — capacity turns
into a structural decision (promote here or one level in), never into a cost.

[AKG](https://doi.org/10.1145/3453483.3454106) (PLDI 2021) is the most Spyre-relevant of
these: polyhedral compilation for Huawei Ascend NPUs, with explicit multi-level buffering
into hardware-managed-by-software buffers, tiling driven by buffer capacity, and the
double-buffering rule written as "≤ half the buffer capacity" — the exact encoding §7.1
argues torch-spyre needs. AKG also reports the compile-time discipline that Q9 needs:
scheduling options in isl were hand-tuned to keep compilation under a minute for 99% of
cases. Around it sit [Tiramisu](https://arxiv.org/pdf/1804.10694v3) (CGO'19),
[Tensor Comprehensions](https://arxiv.org/pdf/1802.04730) (archived in 2023, which is
itself a data point about this approach's industrial trajectory), and
[Diesel](https://dl.acm.org/doi/10.1145/3211346.3211354) — which is a **MAPL'18 six-page
workshop paper**, not a PLDI paper, and whose full text is paywalled, so no number or
mechanism from it should be cited (§15).

One polyhedral result deserves to be quoted in every discussion of model accuracy:
[Prajapati et al.](https://www.pollylabs.org/publications/grosser-2017-Simple-Accurate-Analytical-Time-Modeling-and-Optimal-Tile-Size-Selection-for-GPGPU-Stencils.pdf)
(PPoPP'17) built an analytical time model for GPGPU stencils whose global RMSE exceeds
100% but whose error **near the optimum is under 10%**, which is why it selects tile sizes
well. That is the earliest clean statement of the thesis §8.5 and §13 build on: accuracy
near the optimum is the product, and it is separately testable from global accuracy.

Finally, the honest limit of this line, from Triton's own
[related-work page](https://triton-lang.org/main/programming-guide/chapter-2/related-work.html):
polyhedral methods have "still to be successfully applied to sparse — or even
structured-sparse — neural networks."

### 7.6 Who optimises a real objective — and the gap that survives

| System | Decision | Objective | Granularity |
|---|---|---|---|
| TelaMalloc / MiniMalloc | Placement | Feasibility + compile time | Whole-tensor |
| Timeloop / CoSA / TileLoom | Tiling + mapping | Capacity as filter, then proxy cost | Per-op mapping |
| Welder / Ladder | Tile shape + connect level | Modelled traffic, capacity as filter | Tile-graph |
| Stream / DeFiNES / TileFlow | Fused tiling + core mapping | Latency and energy (simulated) | Fused layer stack |
| [Elk](https://arxiv.org/abs/2507.11506) | SRAM split exec/preload | Latency, capacity as *variable* | Per-core |
| [COSMA](https://arxiv.org/abs/2311.18246) | Schedule + alloc + spill | **Bytes** of non-compulsory traffic | Whole-tensor |
| [OnSRAM](https://dl.acm.org/doi/10.1145/3530909) | Retention/eviction | "Size, liveness, significance" heuristic | Inter-op |
| [Pin or Fuse?](https://dl.acm.org/doi/10.1145/3579990.3580017) | Pin vs fuse | **Inference latency** under mem bound | Layer chain |
| [XLA MSA](https://github.com/openxla/xla/tree/main/xla/service/memory_space_assignment) | VMEM residency + prefetch/evict | **Elapsed seconds** | Whole-tensor |
| [Checkmate](https://arxiv.org/abs/1910.02653) / [Moccasin](https://arxiv.org/abs/2304.14463) | Rematerialization | **Total runtime**, profile-based | Per-op |

OnSRAM is notable for lineage and scale: it comes from the IBM RAPID line and its modeled
system — 3 TFLOP, **2 MB scratchpad**, 32 GB/s external bandwidth — is essentially a Spyre
core; it reports 1.02–4.8× latency reduction (static variant) over no management. "Pin or
Fuse?" (CGO'23) is the classical accelerator paper optimizing the right objective, cutting
feature-map off-chip traffic 50% and latency 15% on a commercial NPU — and it is direct
evidence that fusion alone is not always the right answer, agreeing with FFM, FuseFlow and
DeFiNES.

**XLA MemorySpaceAssignment is the existence proof for a latency objective.** It greedily
promotes HLO values into "alternate memory" (VMEM) ranked by memory-boundedness, where
`CostAnalysis` computes elapsed times *in seconds* (`GetInstructionElapsed`,
`GetInstructionElapsedInAlternateMemory`, `GetAsyncCopyElapsed`), and
`GetAlternateMemoryBenefit` is documented as "how much putting this tensor to the alternate
memory would help if the op is memory bound, or otherwise how far off is the op to memory
boundedness," with large-buffer scaling and a heap simulator for fragmentation. It has real
prefetch *and* eviction. This is the template a torch-spyre LX planner could compute from
its existing per-op model — with the caveat that MSA assumes a single shared scratchpad,
not 32 per-core ones with work division already fixed.

Two tools worth adding to the box.
[Orojenesis](https://people.csail.mit.edu/emer/media/papers/2024.06.isca.orojenesis.pdf)
(ISCA'24) computes "ski-slope" curves of minimum achievable data movement versus buffer
capacity that no mapping can beat, *including* fused operator chains — exactly the curve
needed to answer "is 2 MiB/core the knee for this attention chain?", and DeFiNES supplies
the fused arm of the same experiment nearly for free (Q6). And
[DOSA](https://arxiv.org/html/2509.10702v1) builds a closed-form differentiable analytical
model (0.18% MAE against Timeloop over 10,000 mappings) augmented with a DNN trained on
FireSim RTL data to correct analytical-to-real deviation, then does gradient descent over
~20 tiling variables per layer — a blueprint both for making a cost model differentiable
and for the hybrid analytical-plus-measured-residual structure.

**The gap that survives the correction.** Not "nobody co-optimises tiling and allocation" —
they do. Three narrower things are genuinely unclaimed. (i) **Capacity is never priced.**
Welder assigns infinite penalty above capacity, DeFiNES picks the lowest level that fits,
Stream writes a hard inequality, PPCG refuses promotion, MLIR recurses inward, TileLoom
discards. None models the cost *gradient* near capacity, and none feeds allocator outcome —
spill, fragmentation, solver failure — back into tile selection. Elk is the closest and it
does not fuse. (ii) **Per-core partitioning breaks the standard decomposition.** Welder's
inter-layer independence and XLA MSA's and Checkmate's single-shared-budget assumption both
assume one reuse level with one capacity; with 32 unequal per-core partitions, effective
capacity at the reuse layer is a function of the work division. Untested. (iii) **Fixed
work division is the unstudied regime.** Stream, T10 and Elk all co-optimise the partition,
which Spyre fixes upstream — leaving per-core tile shape × LX residency × fusion depth as a
smaller, better-posed problem with no published capacity-versus-traffic curve. Add that
every objective above is simulated or profile-selected; a *measured-calibrated analytical*
objective is not represented at all.

---

## 8. Tile-size search, mapping canon, and ranking methodology

### 8.1 Three camps, and only one fits a remote, opaque backend

| Camp | Representatives | Cost per operator | Fit for a remote, opaque backend |
|---|---|---|---|
| Measurement + learned surrogate | AutoTVM, Ansor, Helion, `triton.autotune` | 0.65–2.17 h (Ansor); 586 s / 1520 cfgs (Helion) | Poor |
| Construction by alignment or constraints | Roller, Hidet, Bolt, Heron | 0.43 s top-1 (Roller); <1 min enumerate (Hidet) | **Good** |
| Analytical rank → measure top-k | tritonBLAS, TileSight, nvMatmulHeuristics, Pruner, Nautilus | 50–80 µs to rank; k measurements | **Good** |

[AutoTVM](https://arxiv.org/abs/1805.08166) established the template — gradient-boosted
trees over loop-AST features (memory access count, data reuse ratio, vectorize/unroll/
thread-bind annotations) driven by simulated annealing, with hardware in the loop — and
already found that a **rank** loss matches or beats a regression loss. Ansor removed the
templates by sampling complete programs, using 1,000 measurement trials per operator.
Roller measured Ansor at 0.66 h per operator on average, up to 2.17 h. Hidet reports 8–15 h
to tune a single CNN; Bolt cites 7 days for ResNet-50. Modern instances are the same idea
with more measurement: Helion evaluates 1,520 Triton configs in 586 s by differential
evolution and reaches 3.27× geomean over eager on B200 (vs 2.70× for `torch.compile`
max-autotune and 1.76× for hand-written Triton).

**[Roller](https://www.usenix.org/conference/osdi22/presentation/zhu)** (OSDI'22; also
indexed at
[Microsoft Research](https://www.microsoft.com/en-us/research/publication/roller-fast-and-efficient-tensor-compilation-for-deep-learning/))
is the canonical construction reference and the closest published system to what
torch-spyre should build. *It is not an OSDI'22 best paper — the draft said so and was
wrong; the argument does not need it.* Once a tile shape is *aligned* to the hardware's
quanta (memory transaction length, bank count, minimum schedulable unit), the space
collapses and performance becomes predictable. An rTile is built recursively: start small,
at each step enlarge along the axis with the highest data-reuse score
`S_i = (Q(T) − Q(T′_i)) / (F(T′_i) − F(T))` where Q is traffic and F is footprint; stop
when `MemPerf(T′) > MaxComputePerf(T′)` or the level's capacity is exceeded; descend a
memory level; then scale out by replicating the single-core program. The decisive sentence
is: *"the micro-performance model only needs to be accurate when the tile shapes are fully
aligned."* You do not need a globally accurate model; you need one accurate on a
constrained subspace you chose — the same thesis Prajapati reached from the polyhedral side
(§7.5). Results: 0.43 s to emit top-1 against 0.65 h for TVM, within 10% of cuDNN/cuBLAS on
81.5% of 119 operators. And the result that reads like torch-spyre's situation: on
**Graphcore IPU** — immature accelerator, slow opaque device compiler — Roller beat the
vendor library PopART by 3.1× average, up to 9.2×, and matched Ansor (+2.9%) while Ansor
needed hours because IPU compilation alone takes minutes.

[Hidet](https://arxiv.org/abs/2210.09603) (ASPLOS'23) attacks the space from the other
side: because AutoTVM/Ansor tile only by *perfect factors* of the loop extent, their space
is input-shape-dependent, explodes to ~10⁸ schedules for one ResNet-50 conv, and cannot
schedule M = N = K = 2039 at all (2039 is prime). Predicated loading plus a
*hardware-centric* tile space gives fewer than 200 schedules for matmul — ~10⁵ smaller — so
"simply enumerating all schedules would be enough and can be done within one minute."
Tuning time drops 20×/11× vs AutoTVM/Ansor. That prime-shape failure mode matters for
ragged sequence lengths and head dims, and it connects directly to §9's padding question.
[Bolt](https://proceedings.mlsys.org/paper_files/paper/2022/file/1f8053a67ec8e0b57455713cefdd8218-Paper.pdf)
(MLSys'22) reaches the same place through CUTLASS templates plus hardware guidelines —
"tens" of parameter combinations per architecture, pre-generated and profiled — for 2.5×
over Ansor, 300 TFLOPS FP16 GEMM on A100 (>95% of theoretical), whole CNNs tuned in 20
minutes. Bolt also supplies the cautionary number for the measurement camp: **Ansor reached
only 20% of cuBLAS on FP16 GEMM on a T4.**

[Heron](https://dl.acm.org/doi/10.1145/3582016.3582061) (ASPLOS'23) is the third
construction answer and the one the draft missed: rather than sampling a space and
filtering, it *constructs* the schedule space from hardware constraints so that generated
candidates are legal by construction, then searches inside it. It is the natural formal
partner to Roller's alignment argument and to CoSA's in-MIP capacity constraint. ACM DL
returns 403 for the full text; the abstract and its two speedup figures come from the
Semantic Scholar record and its evaluation section was not read (§15).

### 8.2 How well do analytical models actually rank?

Better than most people assume, with a specific failure mode.
[tritonBLAS](https://arxiv.org/abs/2512.04226) (AMD, Dec 2025) is the cleanest experiment
ever run on this question: a purely analytical model, zero autotuning, evaluated against
**exhaustive** measurement over **150,000 random GEMM shapes on MI300X**, achieving
**94.7% selection efficiency** (the analytically-chosen tile lands within ~5% of the
measured oracle) while selecting in 50–80 µs versus 11.9 s–1,383 s for Triton autotune over
75 configs. The residual is not uniform: efficiency degrades at *low arithmetic intensity*,
the latency-bound regime. That matches Roller, whose 27.7%/19.3% losses were "mainly small
operators or with irregular tensor shapes."
[TileSight](https://arxiv.org/html/2607.22432v1) reports 12.35% pooled MAPE on 703 GEMM
shapes across A100/H200/B200/B6000 (against 33.85% for roofline) and shows that keeping the
predicted top 5% prunes 95% of the space while retaining 99.66% of exhaustive-search best on
LLaMA GEMMs. NVIDIA has productized the pattern:
[nvMatmulHeuristics](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/heuristics.html)
"ranks GEMM kernels by estimated performance," with
`CUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM` capping how many ranked candidates get
built and profiled. Triton already has the hook:
[`triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
exposes `prune_configs_by` with a `perf_model` and `top_k` — and in practice almost nobody
supplies a `perf_model`.

Two limits on how far these results carry. Both are GPU-only, and both measure selection
quality on a *mature* backend whose device compiler introduces little variance — which is
exactly the assumption Roller's IPU section and LENS's 493% NPU result say does not hold
here. That is the gap Q1 is scoped to, narrowed by the TpuGraphs/Kaufman correction: rank
correlation against measured latency behind a proprietary backend has been published, with
*learned* models on TPU; an *analytical* model on an explicit-NoC dataflow chip has not.

### 8.3 The mapping canon, and the envelope it prices

The accelerator-mapping community answered "how do you cost a mapping without running it"
a decade before the tile-DSL community asked. Four papers are the canon, and the draft
skipped three of them.

[MAESTRO](https://arxiv.org/abs/1805.02566) (MICRO'19) supplies the vocabulary the rest of
this section reuses: a data-centric notation in which a mapping is described by
temporal/spatial mapping directives over dataflow, and reuse — multicast, spatial and
temporal — is derived from the description rather than pattern-matched. Cite its accuracy
carefully: the paper contains two non-interchangeable claims, "within 3.9% absolute error"
(§4.5) and "within 90–95% accuracy of actual open-source RTL" (introduction), and which one
applies depends on the experiment (§15). Timeloop (§7.3) is the substrate most of the rest
of this canon is built on or validated against, and Accelergy supplies its energy side.

The shared caveat is one this review has to state plainly because it bounds everything
built on top: this canon prices a very narrow object. A single Einsum, at most two input tensors and one output, indices coupled in at most two affine dimensions,
uniformly-distributed sparsity only, no fusion.

**Mind Mappings** ([ASPLOS'21](https://www.kartikhegde.net/media/Mind_Mappings_ASPLOS2021_CR.pdf))
matters less for its headline (1.40×/1.76×/1.29× better EDP than simulated
annealing/GA/RL iso-iteration; 3.16×/4.19×/2.90× iso-time; within 5.3× of a possibly
unachievable lower bound over a ~10²⁵ map space) than for two methodology findings.
First, they train *one* surrogate per algorithm family that generalises across problem
shapes, not per problem — a 9-layer MLP [64,256,1024,2048,2048,1024,256,64,12], 10M
Timeloop-labelled samples, 35 MB, with search quality barely degrading below 5M samples.
Second, and directly relevant to the outlier work now under way: **MSE loss performed
poorly**, because it over-punishes outliers and destabilises training; MAE under-punishes
small variations; **Huber loss won**. Any team refitting because of ≥10% outliers should
treat the loss function as a measured decision, not a default.

**Three papers give three answers to "make search tractable."**
[Sparseloop](https://arxiv.org/abs/2205.05826) (MICRO'22) decouples the model into
independently-evaluable aspects — dataflow, then sparse features, then microarchitecture
— each tractable: >2000× faster than cycle-level, 0.1–8% average error, >99% accuracy on
total MobileNet cycle counts. It advertises "maintains relative performance trends" as a
*separate* claim from absolute error, which is the reporting convention this review
recommends adopting. [FAST](https://arxiv.org/abs/2105.12842) (ASPLOS'22) supplies the
most useful calibration datum anywhere in this area: **Google's internal TPU performance
simulator is within 8.2 ± 2.7% of profiled TPU-v3**, and because it is *optimistically*
biased the authors deliberately baselined against a simulated rather than measured
TPU-v3 so the bias could not be laundered into the result. FAST also puts fusion inside
the search loop as an ILP minimising modelled cycles subject to leftover on-chip
capacity, with weight pinning as a decision variable (space O(10²³⁰⁰); 3.7× single-workload
and 2.4× multi-workload Perf/TDP over TPU-v3), motivated by EfficientNet's unfused
operational intensity of 13–35 FLOP/B against the 208 FLOP/B needed to avoid a bandwidth
bottleneck. Heron (§8.1) is the third answer.

### 8.4 Latency-prediction methodology

The draft jumped from TenSet straight to its own scoring plan. Three papers sit in
between and each supplies something the plan needs.

[nn-Meter](https://air.tsinghua.edu.cn/pdf/nn-Meter-Towards-Accurate-Latency-Prediction-of-Deep-Learning-Model-Inference-on-Diverse-Edge-Devices.pdf)
(MobiSys'21) is the source of the **metric convention**: report RMSE, RMSPE, and "±10%
accuracy" — the *fraction of models* predicted within 10% relative error — over 26,000
models (99.0% mobile CPU, 99.1% Adreno 640, 83.4% Intel VPU). Its mechanism is the real
lesson: predict at the granularity of the *kernel the runtime actually emits*, found by
automated kernel-detection test cases, because operator-granularity prediction collapses
to **8.5%** on the VPU where kernel-granularity reaches 83.4%. For Spyre that granularity
is the DeepTools SuperDSC, not the ATen op.

[Habitat](https://arxiv.org/abs/2102.00527) (ATC'21) is startlingly close to
torch-spyre's own model. Wave scaling predicts
`T_d = (D_o/D_d)^γ (W_o/W_d)^(1−γ) (C_o/C_d)^(1−γ) T_o`, where γ ∈ [0,1] is the kernel's
memory-bandwidth-boundedness **selected from its arithmetic intensity via the roofline
model** — published precedent for making γ a *function* rather than a fitted constant.
Equally important is its error structure: 11.8% average **end-to-end iteration** error,
but 18.0% for MLP-predicted operations and **29.8% for wave-scaled operations**, the gap
absorbed by importance weighting and cancellation. Quoting 11.8% as an operator-level
accuracy is wrong, and a review should insist both per-op and importance-weighted
end-to-end error be reported separately.

[TLP](https://arxiv.org/abs/2211.03578) (ASPLOS'23) closes the loop on rank. It extracts
features from schedule primitives as a language rather than hand-engineered features
(Ansor hand-extracts 164; the TIRAMISU cost model 2534) and evaluates with **both**
dataset metrics (top-1/top-5 score — the achieved-latency ratio of the program the model
ranks first, weighted by subgraph frequency) *and* search metrics. Swapping MSE for
λ-rank loss moves top-1 only 0.9128 → 0.9194, yet the system delivers 9.1×/3.0× CPU/GPU
search-time speedup, and MTL-TLP reaches 4.7×/2.9× using **only 7% of target-hardware
data** — the transfer-learning template for a backend whose hardware is reachable only
on a separate run machine.

### 8.5 The metric to score your model on is not RMSE

[TenSet](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf)
(NeurIPS'21, 52M measured records) is decisive. In their Table 3, Model #1 has the best
RMSE (0.09) and R² (0.77) but yields 7.89 ms end-to-end; Model #3 has RMSE 7.27 and an
R² of **−1818** — a meaningless regressor — but the best pairwise accuracy (0.89), best
top-5 score (0.96), and the **best final latency (6.39 ms)**. Their conclusion: top-k
score reflects the end-to-end objective; RMSE and R² do not. Realistic targets from their
Table 4: top-1 scores 0.65–0.89, top-5 0.83–0.98. Score a Spyre cost model by top-1/top-5
score and pairwise accuracy over candidate tilings *of the same kernel*, following TLP's
definition and nn-Meter's ±10% reporting convention.

### 8.6 Nobody trusts top-1 alone

Roller emits **top-10**, taking average compilation to 13.3 s on V100 and 7.69 s on MI50,
explicitly "to tolerate some hidden performance impacts from device compilers."
[Pruner](https://arxiv.org/abs/2402.02361) (ASPLOS'25) formalizes draft-then-verify: a
cheap symbol-based analyzer drafts, the learned model verifies, giving 2.6–2.7× over
Ansor and 4.08× over MetaSchedule on tensor cores. Nautilus prunes to schedule seeds in
under a minute then spends 256 empirical measurements. With an opaque
static-SuperDSC-to-binary backend, torch-spyre is strictly closer to Roller's IPU case
than its V100 case: the model's job is to hand 5–20 candidates to the run machine, not to
pick one.

Three force-multipliers, one of them upstream. A persisted, keyed autotune cache — IBM's
own [triton-dejavu](https://github.com/IBM/triton-dejavu) exists because
measurement-based tile search "causes high variance in latency" that is "unacceptable for
serving applications in production." Warm-starting from a previous op's optimum
(§8.7, Autocomp). And **Inductor already ships the measure-and-persist pattern**:
`config.runtime_estimations_mms_benchmark` plus `get_estimate_runtime_cache()`
(`scheduler.py:1295–1372`) benchmarks `mm`/`bmm`/`addmm` on device (5 warmup, 10 iters,
10 s cap) and persists the result keyed by kernel name and operand shapes, with
`runtime_estimations_use_nccl_lib_estimations` using vendor-library estimates as anchors.
That is LENS's "measured anchors" pattern already inside the framework and is a better
citation than triton-dejavu alone.

### 8.7 The dataflow mapper literature, and its circularity

[Demystifying Map Space Exploration for NPUs](https://arxiv.org/abs/2210.03731)
(IISWC'22) quantifies the axes: the map space is O(10²⁴) per layer with a
three-orders-of-magnitude best-to-worst spread; **mutate-tile has the highest impact on
EDP** of the three axes; loop order is nearly flat (all 7! = 5,040 permutations collapse
to only 16 distinct EDP values, 14.4× best-to-worst); warm-starting from a previous
layer's optimum converges 3.3–7.3× faster. That is empirical justification for spending
effort on tile size rather than loop order. **Caveat to flag loudly: this entire
literature scores mappings inside Timeloop/Sparseloop — against an analytical model,
never against silicon.** Its answer to "how well does an analytical model rank?" is
circular. tritonBLAS and TileSight validate ranking against real measurement at scale and
are GPU-only; TileFlow (5.4% vs RTL over 131 hand-written mappings), DeFiNES (3% vs
DepFiN), Stream (96–97% vs three chips), ZigZag (5–7.5% vs Eyeriss/ENVISION) and FAST
(8.2 ± 2.7% simulator-vs-TPU-v3) are the accuracy anchors for the accelerator side, and
onnx-mlir's r² table (§6.2) is the anchor for stick hardware specifically.

One recent exception is worth reading:
[Autocomp](https://arxiv.org/abs/2505.18574) runs a two-phase LLM beam search (plan one
optimization, then apply it) with cycle-accurate measurement of every candidate on
Gemmini — a *systolic* accelerator with a 256 KB scratchpad and 64 KB accumulator —
producing code 5.6× faster than the vendor library on GEMM, 2.7× on convolution, and
1.4×/1.1× faster than expert hand-written Exo code. Its schedule-reuse result (5.0× at
~10% of the search budget) is the strongest argument for a warm-start cache keyed on op
shape, independent of search strategy.

---

## 9. Static shapes, bucketing, and the serving objective

The draft contained no serving-level objective at all, and treated dynamic shapes as out
of scope because Spyre is a static-shape accelerator. The literature says the opposite:
**static shapes are exactly what makes bucket and pad selection a compiler cost-model
decision.** LENS's "latency is a step function because of bucketing" was the draft's only
hint and it was not followed.

### 9.1 Dynamic shapes as a cost-model problem

[DietCode](https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html)
(MLSys'22) is the canonical statement and is startlingly close to torch-spyre's
situation. Its cost model is explicitly **multiplicative**:
`Cost_M(P) = f_MK(FeatureExtractor(M)) · f_OCC(P/M) · f_pad(P,M)`, where M is a
micro-kernel (a tile of the complete program P), `f_MK` is a learned cost over the
micro-kernel's features, and the two adaption terms account for core occupancy and
padding waste. The occupancy term is a linear model over the *quantisation of work to
cores*: `f_OCC(P/M) = k·(P/M)/ceil_by(P/M, NumCores) + b`, with `b = 1 − k` so a
perfectly-filled machine costs 1. That is literally the 32-core work-division ragged edge
written as a cost-model factor. Two corollaries for the torch-spyre model: padding waste
should not be modelled as an additive or purely proportional term over padded elements,
because padding and core-count quantisation are separate composing factors; and
**in-kernel boundary checks are not a cheap way to handle ragged tiles** — DietCode
measures up to **17×** degradation on a T4, and its fix is *local padding*: pad the local
workspace on fetch from global memory and slice on writeback, keeping boundary checks only
at the memory stages where they hide under transfer latency and removing them from the
compute stage. For an explicitly-staged LX scratchpad that is the natural strategy and it
is cheap to express. Payoff: 5.88× less auto-scheduling time on 8 sampled sequence lengths
(94.1× projected over [1,128]), up to 69.5% lower latency than Ansor and 18.6% than the
vendor library on BERT-base (29.9% / 5.4% average). Baseline pain: Ansor needs ~42
CPU-hours to tune one dynamic-shape workload. Dispatch is a decision tree trained offline
over shape→micro-kernel votes and emitted as C.

[Nimble](https://arxiv.org/abs/2006.03031) (MLSys'21) is the prior generation and is
instructive mainly as a *negative control* DietCode measures against: tuning the largest
shape and reusing that schedule everywhere with loop partitioning collapses when tiles are
small (351% worse on BatchMatmul NT, because loop partitioning needs t ≪ T to pay off).
Its durable contributions are the `Any` dimension with gradual typing, runtime shape
functions, and **symbolic codegen by residue**: for a symbolic dim tiled by 8, generate 8
kernel copies specialised to `x = 8k + r`, dispatching on the residue — full dispatch
matches static-shape performance and reducing the kernel count degrades it monotonically.
Its memory planning cut buffer allocations 47% and allocation latency 75% for ≤8% extra
footprint. [Relax](https://arxiv.org/abs/2311.02103) (ASPLOS'25) replaces `Any` with
first-class **symbolic** shape annotations so relations between dims survive
optimization, and shows the benefit that matters most for a scratchpad machine: planning
memory against symbolic upper bounds allows fully static pre-allocation for
dynamic-shaped tensors, cutting activation memory **22%** (prefill) and **40%** (decode)
on Llama3-8B and enabling CUDA-graph capture that would otherwise be impossible. That is
directly applicable to `LX_PLANNING` and `HBM_POOL_PLANNING`.
[SoD²](https://arxiv.org/abs/2403.00176) (ASPLOS'24) generalises the static side:
classifying 150 ONNX operators into four dynamism degrees and running forward/backward
Rank-and-Dimension Propagation over a lattice of known/symbolic/derived constants yields
27–88% memory savings and 1.7–3.9× speedups over ORT, MNN, TVM+Nimble and TFLite. Its
allocation plan lands at **1.05× optimal peak memory versus 1.16× for greedy** — evidence
that shape-inference precision converts directly into scratchpad bytes — and its
multi-version codegen for hotspot GEMM/CONV alone contributes 1.3–1.6% (CPU) and 1.4–1.7×
(GPU), i.e. bucketing is a first-class optimisation, not a fallback.

### 9.2 How big is a bucket set? Production sets the bounds

[AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html)
— the closest deployed analogue to Spyre — auto-buckets at powers of two from 128 up to
max context length (context encoding) and max sequence length (token generation),
dispatches to the **smallest bucket that fits**, is bounded by ~**2 GB of DRAM per
NeuronCore**, and warns that two-dimensional bucketing with prefix caching "exponentially
increases the number of context encoding buckets."
[vLLM](https://docs.vllm.ai/en/stable/design/cuda_graphs/) sits at the other end: its
default CUDA-graph capture set is `[1,2,4] + range(8,256,8) + range(256, max+1, 16)`,
capped at 512 by default (1024 on data-center Blackwell) — **51 buckets** — with inputs
padded up to the nearest captured size and a graph usable only at the size it was
captured for. So the practical design space runs from roughly 5 to roughly 50 compiled
variants, and the binding constraint is **compiled-artifact memory, not compile time**.
That is a constrained optimisation the cost model could solve directly: minimise
`Σ_s p(s)·Cost(bucket(s))` over a shape distribution subject to a memory budget. Nobody
in this literature does it with a *calibrated analytical* model rather than a learned one.

### 9.3 The objective function comes from vllm-spyre / sendnn-inference

The plugin — repository [vllm-spyre](https://github.com/vllm-project/vllm-spyre), now
`torch-spyre/sendnn-inference`, package `sendnn-inference`, `VLLM_PLUGINS=sendnn_inference`,
env prefix `SENDNN_INFERENCE_*` with `VLLM_SPYRE_*` documented as legacy — defines the
unit the cost model must actually price. Cite both name sets or the review reads as stale.

`sendnn_inference/v1/worker/spyre_worker.py` compiles a **prefill graph and a separate
decode graph** and eagerly deploys both, commenting that this ensures "the compiled decode
program is also installed on the device before runtime" so "first runtime decode does not
pay a one-time deploy cost (observed as elevated ITL on the very first decoded token)."
Pooling models loop over prompt/batch-size combinations, logging
`[WARMUP] (%d/%d) for prompt length %d with batch size %d`, with total warmup reported as
`[WARMUP] Finished in %.3fs (compilation cache %s)`. Per the
[configuration guide](https://docs.vllm.ai/projects/spyre/en/latest/user_guide/configuration.html),
generative models use **chunked prefill** with `max_num_batched_tokens` recommended
512–4096 and required to be a multiple of the fixed **block size 64** (auto-set to 512 for
`ibm-granite/granite-3.3-8b-instruct` at tp=4), plus prefix caching on by default; pooling
models use **static batching**, one graph precompiled per
(`SENDNN_INFERENCE_WARMUP_PROMPT_LENS`, `SENDNN_INFERENCE_WARMUP_BATCH_SIZES`) pair
(legacy `VLLM_SPYRE_WARMUP_PROMPT_LENS`/`_BATCH_SIZES`/`_NEW_TOKENS`), with requests
outside the compiled set **rejected** and "no up-front checks" that a compiled graph fits
memory. `TORCH_SENDNN_CACHE_ENABLE`, `TORCH_SENDNN_CACHE_DIR` and
`SENDNN_INFERENCE_REQUIRE_PRECOMPILED_DECODERS` exist because "model compilation can be
resource intensive and disruptive in production environments."

The [supported-features matrix](https://docs.vllm.ai/projects/spyre/en/latest/user_guide/supported_features.html)
bounds the objective further: chunked prefill, prefix caching, guided decoding, beam
search, LogProbs, tensor parallel and embedding models are fully supported; quantization
and multi-modality are experimental; LoRA, speculative decoding, encoder-decoder,
pipeline/expert/data parallel, **prefill-decode disaggregation** and sleep mode are "not
planned." No disaggregation and no speculative decoding means TTFT and TPOT must be
optimised on the same device with the same compiled artefacts — the cost model cannot
assume the two phases are independently tunable. The original
[RFC #9652](https://github.com/vllm-project/vllm/issues/9652) explains why bucketing
exists at all: all compilation is triggered at initialisation so it never lands on the
critical path, with a warmup routine compiling every needed (prompt length, output tokens,
batch size) shape "similar to CUDA graph behavior"; phase 1 mirrored AWS Inferentia's
Executor/Worker/ModelRunner split and phase 2 forbade prefilling new requests "until all
decodes in the running batch are finished" — a scheduler-level restriction, useful for
separating how much of the bucketing burden is hardware from how much is historical.

**So the served objective is: minimise TTFT over a chunked prefill graph and TPOT/ITL over
a block-64 paged-KV decode graph, across a chosen finite set of shape buckets, subject to
a compile-time budget spent per bucket.** The budget is real — onnx-mlir's BERT-base
compile is 174.10 s (18.6% MLIR, 48.3% LLVM opt, 31.4% llc), reduced to 139.01 s by
exporting constants to an external file before invoking LLVM tools — and the roadmap's
"few minutes per priority model" is the local version of it. Bucket-set selection is the
cost model's missing consumer: it should score a *candidate bucket set* (padding waste ×
arrival distribution × compile cost), not just a single loop nest.

---

## 10. Block-scaled quantization is a layout problem

The roadmap says FP8 and the draft has zero coverage. The key framing: a low-bit format
is not merely a narrower dtype that leaves stick alignment untouched.

**The formats.** [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
fixes MXFP8/6/4 at **block size 32** with a shared 8-bit E8M0 scale (an unsigned biased
Float32 exponent, no Inf, single NaN encoding); an MXFP4 block of 32 elements occupies 17
bytes. NVFP4 halves the block to 16, uses an E4M3 first-level scale, and adds a per-tensor
FP32 second level.

**The compiler consequences are concrete and hostile to naive layout passes.**
[Transformer Engine's MXFP8 docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)
require the **last dimension divisible by 32 and the product of all other dimensions
divisible by 32**, and state that row-wise (1×32) and column-wise (32×1) quantizations
"cannot derive one from the other. Both must be quantized independently from the
full-precision data" — i.e. **a block-scaled tensor has no free transpose**, so any pass
that reorders axes must re-quantize or carry two copies. Scales themselves need
hardware-specific swizzling: 128×4 vertical slices linearised into the interleaved order
0, 32, 64, 96, 1, 33, 65, 97, … [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
requires the LHS scale factor to be "TMA-aligned and transposed," with SM100 packing four
UE8M0 values into one int32 over 1×128 activation and 128×128 weight blocks (up to 1550
TFLOPS on H800), and JITs shapes and block sizes as compile-time constants.
[MARLIN](https://arxiv.org/abs/2408.11743) reorganises group-128 scales offline "such that
the scales required by the same type of thread, for different 16×16 blocks, are packed
together" to permit 16-byte vector loads, plus an XOR swizzle `i⊕j` for bank-conflict-free
access — reaching 3.87× over FP16 at batch 1–16, 3.5× at 32, 2.0× at 64, 1.5× at 128, with
a roofline argument (FLOPs/byte ≈ 200 on an A10) putting the compute-bound crossover at
b_opt ≈ 50, and 2.93×/2.74×/1.78× end-to-end in vLLM on Llama-2-7B at batch 1/16/64.
[QServe](https://arxiv.org/abs/2405.04532) (MLSys'25) reports that existing INT4 methods
pay **20–90% runtime dequantization overhead** and recovers 1.2×/1.4× (Llama-3-8B,
A100/L40S) and 2.4×/3.5× (Qwen1.5-72B) over TensorRT-LLM via compute-aware weight
reordering. **Ladder** (§7.4) is the only work treating this as a *general compilation*
problem, with `tType` — a tile-wise type carrying width, element shape and conversion
functions — transformed across memory layers, reporting ~2.3× average over vLLM (companion
implementation BitBLAS: W4A16, W2A16, W1A16, NF4, FP4/FP8 E2M1/E4M3/E5M2, INT8×INT1/2/4).
[torchao](https://arxiv.org/abs/2507.16099) supplies the PyTorch-native mechanism a Spyre
backend would extend: quantization via tensor subclasses so `nn.Linear` stays
`nn.Linear`, with `AffineQuantizedTensor`'s external shape/dtype decoupled from an
internal `tensor_impl` holding either plain `(int_data, scale, zero_point)` or a
device-specific packed format, `block_size` controlling granularity, and a **`Layout`
abstraction** (e.g. `TensorCoreTiledLayout(inner_k_tiles=8)`) selecting the physical
packing. A Spyre stick-aligned packing of quantized weights plus their scale tensor would
live in a `Layout` subclass, keeping the packing decision out of the model graph.

**Two framing sources close the loop.** DeepSeek's
[V3 hardware retrospective](https://arxiv.org/abs/2505.09343) (ISCA'25) argues that
fine-grained scaling "introduces large dequantization overhead in transporting the partial
results from Tensor Cores to CUDA Cores for scaling factor multiplication," notes Hopper
keeps only the top 13 fraction bits (FP22 accumulate), and recommends hardware "natively
support fine-grained quantization, enabling Tensor Cores to receive scaling factors." That
states the architectural cost in dataflow terms and poses the exact question for Spyre's
matmul pipeline: does scale application belong inside the DeepTools-generated inner loop
or in a separate pass? And an October 2025
[INT-vs-FP study](https://arxiv.org/abs/2510.25602) challenges the FP-centric default: at
block size 32 the measured 75th-percentile crest factor over 10,752 real tensors is 2.96
(2.39 after Hadamard rotation), below the MXINT8/MXFP8 crossover of 7.55, so **MXINT8
beats MXFP8** on accuracy at 8 bits, while an MMU cost model puts MXINT8 at 0.63× energy
and 0.79× area of MXFP8 and NVINT4 at 0.34× / 0.38× of NVFP4 (mixed MXINT8+NVINT4 versus
MXFP8+NVFP4: 0.75× energy, 0.66× area). Block-scaled *floating point* is not the settled
direction.

**Implication for torch-spyre (my analysis, not a cited claim).** A 128-byte stick is 64
fp16 elements = **2 MX blocks**; at MXFP4 it is 256 elements = **8 MX blocks**, whose
scales occupy 8 bytes — 1/16 of a stick. So the scale tensor's natural granularity is
16–32× finer than the data tensor's stick, and the layout solver will be asked to place a
companion tensor that is stick-*sub*-granular, cannot be transposed independently of its
data, and constrains tile sizes to multiples of 32 along the reduction axis. That interacts
directly with LX planning and with the pad/bucket choice, because a bucket boundary that
is not a multiple of the block size forces re-quantization rather than mere padding. This
is Q8 in §12.

---

## 11. Where torch-spyre sits

Grounded in the repository as of this branch, not in the literature's description of it.

### 11.1 What it already does that is standard

torch-spyre is an out-of-tree Inductor backend registered through the documented
`Scheduling`/`Kernel`/`WrapperCodegen` contract (pytorch#99419). Its
`SpyreHeuristics(InductorChoices)` in `torch_spyre/_inductor/choices.py` returns `False`
from `can_fuse`, `can_fuse_vertical` and `can_fuse_horizontal`, installed via
`V.set_choices_handler(SpyreHeuristics())` at `patches.py:144` — the officially
sanctioned one-kernel-per-op configuration (dev-discuss 3226), not a hack. Fusion is
instead performed backend-side in `fusion.py` (`spyre_fuse_nodes`), bundling contiguous
runs of Spyre nodes. That is legitimate and increasingly common — it is what vLLM does
with custom-op barriers plus its own FX passes, and what TileLang and tt-mlir do by owning
their own tiling layer.

Coarse tiling with user hints and a sequential loop grid (`CountedLoopSchedulerNode`,
`op_it_space_splits`) is essentially Pallas's `BlockSpec` + grid, spelled differently.
Stick alignment (128 B / 64 fp16 elements) is Roller's alignment quantum, Ladder's
`GetDeviceHint` granularity, TPU's 8×128, Trainium's fixed partition dim 128 — and, first
of all, zDNN's own `AIU_BYTES_PER_STICK`. Compile-time liveness-based scratchpad
allocation with multiple interchangeable solvers — `greedy_solver.py`,
`firstfit_bestfit_solver.py`, `ilp_solver_ortools.py`, `simulated_annealing.py`,
`plan_solver.py`, plus `permutation_layout.py` — is precisely the TelaMalloc/MiniMalloc/
Poplar/tt-metal design point, and having an OR-Tools ILP already wired in puts
CoSA-style and Stream-style formulations within reach. Layout propagation and restickify
insertion/optimization (`propagate_layouts.py`, `insert_restickify.py`,
`optimize_restickify.py`) are the onnx-mlir ZHigh problem and the DNNDaSher problem,
already recognized as first-class passes. Work division and core mapping
(`work_division.py`, `core_mapping.py`) are spatial mapping in Timeloop/CoSA/MAESTRO-Cluster
terms, and Rammer is the prior art. HBM pool planning is the second memory level.

And crucially, the analytical cost model is not purely observational: while
`cost_model_pass.py` is a "predicted-runtime reporting" pass over the pre-scheduling
loop-level IR, `work_division.py::cost_model_matmul_division` already uses the model to
*choose* matmul work division across cores. So torch-spyre has exactly one
cost-model-driven decision today — one more than upstream Inductor has for a non-GPU
device, where `_get_estimated_runtime` returns literal zero.

### 11.2 What is standard elsewhere and missing here

**Layout-conversion passes onnx-mlir already ships.** Three specifically: an explicit
conversion-*count* benefit test before firing a decomposition; fusion of the conversion
into the consuming compute op (`FusionOpStickUnstick`); and compile-time stickification of
all constant weights (`ZHighConstPropagation`), which eliminates runtime stickification of
weights entirely. Also `--nnpa-disable-*` style per-pass ablation flags, which is the
harness Q2 needs.

**A layout algebra.** Layout is currently a set of descriptors plus hand-rolled
propagation. Linear Layouts' argument is not primarily speed — ad-hoc layout kinds produce
a quadratic conversion surface and a long bug tail (12% of filed Triton bugs). CuTe,
linear layouts, Axe and Graphene all offer composition, product and division as
operations; onnx-mlir's ZLow shows the cheapest version, the layout as a plain affine map
on a 4 KiB-aligned memref. Nobody has published whether SuperDSC's
`layoutDimOrder_`/`stickDimOrder_`/`stickSize_` scheme is expressible as an affine map or
an Axe layout; that is a well-defined, roughly one-week question.

**Layout choice as search.** tt-mlir enumerates legal layouts per op and beam-searches
(width 8) with backend-validated constraints. torch-spyre propagates; it does not search.

**Restickify-aware *placement*.** onnx-mlir's five-round fixed point charges stick/unstick
per edge inside the placement objective. torch-spyre's model does not yet price layout
conversion inside any placement or fusion decision.

**A candidate-ranking loop.** Every serious cost-model system in §7 and §8 has the same
shape: generate candidates → filter by capacity → rank by model → measure top-k. Roller,
Welder, TileLoom, nvMatmulHeuristics, Pruner, Nautilus and Triton's `perf_model`/`top_k`
hook all instantiate it. torch-spyre has the model but not the loop.

**Fusion of any kind through Inductor.** With `can_fuse` returning `False`, no
FlashAttention-shaped fusion is representable in the Inductor path at all. KTIR is the
declared vehicle for kernel-scoped fusion — but until something occupies that layer, §4's
entire literature is inapplicable. §2 argues the cheapest first step is not a new IR but
schedules-as-searchable-data.

**Residency and spilling with a runtime objective.** LX planning is feasibility, like
everyone else's. XLA MSA and Elk are the counterexamples worth copying.

**Two granularities, not one.** zDNN carries a 4 KiB / 32-stick page above the stick and
the NNPA model's matmul term blends `ceil(·,64)` with `ceil(·,32)` page-level work. The
torch-spyre model currently carries only the stick.

**A double-buffering-aware capacity constraint.** If DeepTools stages tiles by
double-buffering, Buffets says the tiler's effective capacity is half the physical LX.
AKG encodes exactly this as "≤ half the buffer capacity."

**Validation methodology.** RMS error across kernels is the wrong metric (TenSet).
Top-1/top-5 score, pairwise accuracy and Kendall τ over candidates of the same kernel are
the right ones (TLP), reported alongside a ±10% fraction (nn-Meter) and separated into
per-op and importance-weighted end-to-end error (Habitat).

**A decided escape-hatch story.** AWS shipped NKI beside the Neuron compiler; OpenAI
shipped Gluon beside Triton; Tenstorrent kept Metalium under TTNN; AMD ships IRON under
MLIR-AIR. Production tile stacks converge on a two-tier design. Whether KTIR is the
automatic layer, the escape hatch, or both is unresolved with precedent on all sides.

**A persisted autotune/warm-start cache**, and **an instruction-level test assertion**
(onnx-mlir's `TEST_INSTRUCTION`) that checks *which* DSC the pass chain produced, not only
that the numerics match.

**A bucket-set consumer.** §9. The cost model prices loop nests; production prices bucket
sets under a compile budget.

### 11.3 What is genuinely novel about the dataflow setting

Three claims survive contact with the new material, and one was withdrawn.

**No published tile IR or layout algebra targets Spyre/AIU as a *tile* machine.** Axe
states it does not yet address dataflow/streaming accelerators. onnx-mlir models the stick
layout but invokes one zDNN primitive at a time, with no multi-core work division, no
scratchpad allocation and no overlap term. No source explains how a layout algebra should
model a DMA/stick descriptor as the atom rather than a thread–value map.

**Layout compatibility as correctness, not performance.** GPU work treats layout as a
performance concern (linear layouts' bug statistics being the partial exception).
DNNDaSher states it as a functional-correctness invariant on this exact hardware family,
with a 2.3× average prize. onnx-mlir encodes the same invariant structurally. That framing
is rare and remains a genuine differentiator.

**Modelling around an opaque, out-of-process backend.** The compute/HBM overlap that a γ
term captures is generated by DeepTools from a static SuperDSC JSON, outside the IR. Three
consequences. (i) The warp-specialization/pipelining literature — Twill, Triton autoWS,
FA3/FA4 — is *descriptive* for torch-spyre, not prescriptive: those knobs are not owned by
this compiler. (ii) First-principles overlap modelling is the known failure mode; LENS is
the clearest published acknowledgement and its answer is measured anchors plus analytical
composition, while XLA's `kMemoryComputeParallelism = 0.95` is the shipping precedent for a
single fitted overlap scalar and Habitat's roofline-selected γ is the precedent for making
it a function of arithmetic intensity. (iii) **Withdrawn:** the draft claimed nobody has
published rank correlation of an analytical model against measured latency behind such a
backend. onnx-mlir's `PerfModelArch14/15.inc` publishes per-op regressions fitted on
measured z16/z17 hardware with r² in-source, driving a production placement pass; and
TpuGraphs ([NeurIPS'23 D&B](https://arxiv.org/abs/2308.13490)) and
[Kaufman et al.](https://arxiv.org/abs/2008.01040) (MLSys'21) score *learned* models by
Kendall τ against measured TPU latency behind the proprietary XLA:TPU backend. The
surviving novelty is narrower and should be stated exactly: **an analytical model, scored
by rank, on an explicit-NoC multi-core scratchpad machine, over a joint tile ×
work-division space.** Not "an analytical model behind an opaque backend."

Add to that a per-core scratchpad structure (32 × 2 MiB with work division already fixed)
that breaks the single-shared-budget assumption in XLA MSA, Checkmate and Welder's
inter-layer independence argument; and — my arithmetic, flagged — a ~4× lower
SRAM-per-unit-of-off-chip-bandwidth ratio than MTIA v2, which if it holds argues for
weighting working-set reduction more heavily than Meta needs to.

### 11.4 The honest summary

torch-spyre has built the asset that two frontier roadmaps name as their next step —
tt-mlir's unimplemented "Cost Mode" and Triton autoWS's "model-based global optimization"
— and that none of the MLIR stacks ships, on hardware where the alternative (cheap
exhaustive measurement) is unavailable, which is exactly the regime where Roller argued a
static model wins. It is *not* the first analytical model for stick hardware; IBM's own
onnx-mlir is, and its published residuals should be read both as a caution (layout
conversion is the hardest term to fit) and as a reassurance (torch-spyre's outliers are
not anomalous). What torch-spyre has not built is the loop that consumes the model, and
what nobody has built is that loop over a *joint* tile × LX × work-division space scored
by measured latency. The gap between "we have a cost model" and "the cost model decides"
is the research program.

---

## 12. Open research questions

Each is scoped for a small team with one accelerator and a remote run machine. Q1–Q6 are
revised; Q7–Q9 are new.

### Q1. Does analytical *ranking* survive an opaque backend, on a NoC dataflow chip?

**Narrowed.** Ranking-versus-measured on a non-GPU accelerator behind a proprietary
compiler is not virgin ground: TpuGraphs has a tile-size configuration collection scored by
Kendall τ against measured TPU latency behind XLA:TPU, Kaufman et al. report ranking
quality on the same setup, and TileLoom's top-2 profiling on Tenstorrent adds only +4.7%.
Those are **learned** models, and TPU is not an explicit-NoC dataflow chip. IBM's
`PerfModelArch15.inc` is analytical and fitted to measured hardware but is scored by
per-op r², not by rank over candidate mappings, and its worst fits (NNPA Unstick 0.429,
Stick 0.691, MatMul_3ds 0.706) tell you where the difficulty is rather than how well the
model ranks. **Surviving novelty:** an *analytical* model, scored by *rank*, over a joint
tile × work-division space, on an explicit-NoC multi-core scratchpad machine.
**Related work.** TpuGraphs, Kaufman et al., tritonBLAS, TileSight, TenSet, TLP, nn-Meter,
Habitat, Roller, LENS, TileLoom, onnx-mlir PerfModelArch15.
**Experiment.** Pick 20–40 kernel shapes spanning the workload mix (dense mm, bmm,
attention chains, MoE expert GEMMs, elementwise chains). For each, enumerate all
stick-aligned, LX-feasible tile/work-division candidates and measure *all* of them on the
run machine (at <1 min/run a few thousand points is affordable). Report Kendall τ, top-1
and top-5 score per kernel — not RMS across kernels — plus the ±10% fraction, and plot each
against arithmetic intensity to locate the degradation regime tritonBLAS and Roller both
report. **Pre-register the repeat/noise protocol before collecting anything**: a rank
metric computed on singleton measurements is the exact failure mode already encountered
once, where a variable-γ trend of r = +0.91 collapsed to +0.05 on repeat-backed data.
Fix repeats per point, drop singletons, and state the minimum detectable rank difference.
Deliverable: the first published rank-quality curve for an analytical model behind a
proprietary dataflow backend, plus a defensible choice of *k*.

### Q2. Which constraint actually binds on LX — capacity or layout legality?

**Genuinely open, cheapest, best-posed. Keep.** tt-mlir found that in real models "spills
are constraint-driven, not memory-driven," with 40–94% L1 headroom remaining even when
spills fired. If that replicates on Spyre, effort spent on smarter LX packing is
misallocated and belongs on layout legality and propagation instead — where DNNDaSher's
2.3× average sits. The two are separately measurable and have never been separated on this
hardware. Two additions since the draft. First, the effective-capacity question is part of
the same measurement: if DeepTools double-buffers, Buffets says the tiler's real budget is
~1 MiB/core and AKG's "≤ half capacity" rule is the encoding. Second, onnx-mlir's
`--disable-zhigh-*` / `--nnpa-disable-fusion-op-stick-unstick` flags are a shipped
per-pass ablation harness worth copying wholesale.
**Related work.** tt-mlir TTNN Optimizer; onnx-mlir ZHigh pass family and its flags;
DNNDaSher; Buffets; AKG; TelaMalloc/MiniMalloc; Interstellar.
**Experiment.** (a) Instrument the LX planner over the existing sweep corpus to record, at
every spill or failure, peak LX occupancy and the *reason* — genuine capacity exhaustion
versus a layout constraint forcing materialization — and report the distribution, split by
whether the effective budget is 2 MiB or 1 MiB. (b) Ablate `optimize_restickify` per pass:
count restickify ops and their modelled and measured cost across the seven priority models,
with and without coarsening, and separately with compile-time stickification of constants
(onnx-mlir's `ZHighConstPropagation`) enabled. This answers "how much of DNNDaSher's 2.3×
is still on the table after Inductor-level layout propagation," and it decides where the
next quarter of engineering goes.

### Q3. Joint tile × LX under a *measured* objective, with work division fixed

**Restated — the draft's version was contradicted.** "Nobody co-optimizes tile shape and
scratchpad allocation" is false: Welder filters candidates by `MemFootprint` against
`level.capacity` *inside* the tile search; Stream's WACO puts per-core capacity `C_j`
directly in the ILP alongside the tiling variable; DeFiNES sweeps tile size against
capacity for fused stacks; TileFlow models 384 KB per core across four cores; Elk
partitions each core's SRAM between execution and preload space; Baskaran et al. and AKG
do it as a constrained NLP. Three things remain genuinely unclaimed. (i) **Capacity is
never priced** — nobody feeds allocator outcome (spill, fragmentation, solver failure) back
into tile selection. (ii) **Welder's inter-layer independence may not survive per-core
partitioning**: when the reuse level is per-core and the 32 partitions are unequal (tail
cores), effective capacity at the reuse layer becomes a function of the work division, so
the decomposition argument does not transfer unmodified — and nobody has tested it.
(iii) **Fixed work division is the unstudied regime**: Stream, T10 and Elk all
co-optimise the partition, which Spyre fixes upstream, leaving per-core tile shape × LX
residency × fusion depth as a smaller problem with no published capacity-versus-traffic
curve. Add that every one of these objectives is either simulated or profile-selected; a
*measured-calibrated analytical* objective is not represented.
**Related work.** Welder, Ladder, Stream, DeFiNES, TileFlow, Elk, Chimera (closed form
`T* = −α + sqrt(α² + MC)`), Roller, CoSA, FFM, TileLoom, COSMA, DOSA, Baskaran'08, AKG.
**Experiment.** On a fixed corpus, compare four formulations end-to-end: (a) current
sequential greedy; (b) Roller-style aligned construction with capacity as the stopping
rule; (c) Chimera's closed form as an analytic baseline; (d) a joint CP-SAT model over tile
factors and LX allocation with capacity as a linear constraint — the OR-Tools machinery is
already present in `scratchpad/ilp_solver_ortools.py`, and Stream's inequality is the
template. Score all four by *measured* latency. Then run the independence test directly:
hold the tile config at one layer fixed, vary the work division, and check whether measured
traffic at that layer moves. Finally, answer TileLoom's unasked question by deliberately
admitting candidates that overflow LX by ≤10% with an explicit spill, to test whether the
hard pre-filter is discarding optima. Report compile time against the roadmap's
"few minutes per model" budget throughout (see Q9).

### Q4. Is the FlashAttention dataflow even the right target on Spyre?

**Open; strengthened.** The GPU literature assumes it is and asks "can we derive it." But
FlatAttention reports 4.1× speedup and 16× less HBM traffic from *replacing* the FA-3
dataflow with on-chip NoC collectives on tile-based hardware; Sohn et al. find the winning
transformation for streaming dataflow is operator reassociation, not tiling; TileFlow
matches FLAT at an order of magnitude lower L1 footprint by tiling the softmax column
dimension FLAT leaves whole; and FFM, FuseFlow and DeFiNES all find that always-fusing is
not optimal and that the optimal fusion set shifts with sequence length. Two additions.
First, **Ring Attention and Blockwise Parallel Transformers belong next to FlatAttention**
— same "keep K/V on-chip and move it around the fabric" idea, one generation earlier, on
GPUs; both are cited from memory here and must be fetched before use (§15). Second, a
**numerics sub-question the draft omitted entirely**: on an fp16-native chip, does the
online-softmax recurrence even hold at the sequence lengths in the seven priority models?
FA4's *numerically*-motivated conditional rescaling (skip the correction unless the new max
threatens stability, cutting corrections ~10×) is outside every algebraic framework in §4,
and DeepSeek's FP22-accumulate observation shows accumulate width is a first-order
architectural fact. Establish the accumulate width of the Spyre vector arrays and the
worst-case rescale magnitude before promising derivation.
**Related work.** FlatAttention, Sohn et al., TileFlow/FLAT, FFM, FuseFlow, DeFiNES,
Neptune, Flashlight, Mirage, Nautilus, Event Tensor, Ring/Blockwise attention (unverified),
FA4/Modal.
**Experiment.** (a) Determine whether the Spyre fabric supports multicast/in-fabric
reduction, or only core-to-core DMA. FlatAttention's entire 16× traffic reduction is
attributable to keeping K/V on-chip via NoC collectives; without in-fabric reduction most
of that win is unavailable and the FFM buffer-allocation angle matters more. Days of
documentation work, and it gates everything else. (b) Implement three attention lowerings —
unfused; an FA-style KV-block loop with online softmax; and a broadcast/collective variant
— and compare predicted against measured across sequence lengths, testing FFM's and
DeFiNES's shared claim that the optimal fusion set moves. Before promising derivation,
enumerate which Spyre-relevant reductions have an invertible `g` in Neptune's sense
(Welford is the paper's stated failure; layernorm/RMSNorm variants need checking).

### Q5. Can a predicted-latency `score_fusion` beat Inductor's byte proxy?

**Reframed as an Inductor-integration question — the general question is answered
elsewhere.** XLA:GPU's PriorityFusion already ranks fusion candidates by predicted runtime
deltas (`Priority = absl::Duration`, `current_priority + time_unfused − time_fused`),
Welder ranks tile-graph candidates by analytic traffic, and Apollo and AStitch rank
fusions under memory constraints — a production compiler has already replaced a heuristic
fusion score with an analytical latency model. What is unanswered is the *specific*
version: **inside Inductor, for a non-GPU backend, where benchmarking is unaffordable.**
Upstream fusion is ranked by summed shared bytes plus tie-breakers with no time quantity
anywhere; the single runtime-estimate-driven fusion decision is dead by default; the one
user-cost-model hook is wired only to comms; `_get_estimated_runtime` returns zero off-GPU;
and pytorch#149603 has been open since March 2025 asking for steerable fusion. Nobody has
upstreamed a `score_fusion` override driven by an analytical model, because on GPUs
empirical benchmarking is cheap enough that nobody needed to. torch-spyre already
subclasses the exact object that owns the hook, and `config.inductor_choices_class`
(`config.py:770`) is the supported installation point.
*The draft's two "determine first" sub-tasks are deleted: the `benchmark_fusion` guard is
literally CPU-and-not-Triton (`scheduler.py:4013–4015`), not a non-Triton proxy — the real
barrier is `BaseScheduling.benchmark_fused_nodes` raising `NotImplementedError` — and the
memory-planning question is answered by reading `codegen/memory_planning.py`.*
**Related work.** XLA PriorityFusion and `gpu_performance_model_base` (including
`kMemoryComputeParallelism = 0.95` as a sanity anchor for γ); Welder; Apollo; AStitch;
DNNFusion's type lattice + targeted profiling; Inductor `choices.py`/`scheduler.py`;
vLLM's ten hand-written passes (which quantify what the byte proxy leaves on the table:
5–20% for AllReduce+RMSNorm alone); QuACK; Flashlight.
**Experiment.** Behind a flag, re-enable Inductor fusion for a restricted op class and
override `score_fusion` to return Δ(predicted cycles) instead of `memory_score`. Build a
corpus of fusable op pairs from the priority models and compare three orderings —
Inductor's byte proxy, the predicted-latency delta, and measured ground truth — by pairwise
accuracy. Adopt XLA's incremental re-prioritisation (re-cost only affected producers) to
keep compile time bounded. A first check that costs nothing: replicate QuACK's finding by
counting global loads per element in the current softmax/layernorm lowerings; if the extra
load is there, that is a pure accounting win visible in the IR before any hardware run. A
cheap complement worth testing alongside: DNNFusion's five-way mapping-type table, where
only ambiguous pairs consult the model at all.

### Q6. Where is the knee of the 2 MiB/core LX capacity curve?

**Open and cheap; extended.** Every decision in Q2–Q4 presupposes an answer to "how much
would more (or less) on-chip capacity actually buy on our workloads," and nobody knows it
for Spyre. Orojenesis computes exactly this for the **unfused** arm — a bound on minimum
data movement that no mapping can beat, as a function of buffer capacity — and DeFiNES
gives the **fused** arm nearly for free, since it already sweeps tile size × overlap mode ×
fuse depth and emits capacity-versus-traffic curves for fused layer stacks, with the
finding that the optimum is always interior and that the top-level memory assignment flips
per tile-type as the working set crosses each level. Interstellar's negative result says
hierarchy sizing dominates dataflow choice, which makes the curve the highest-value single
artifact. Two extensions: run the sweep at both 2 MiB and 1 MiB per core to bracket the
double-buffering question from Q2, and compare the interior optimum against Chimera's
closed form as a sanity check on the analytical path.
**Related work.** Orojenesis, DeFiNES, Stream, Chimera, Buffets, Interstellar, Marvel,
Pin-or-Fuse, OnSRAM (modelled on a 2 MB scratchpad at 3 TFLOP and 32 GB/s — essentially a
Spyre core), COSMA.
**Experiment.** Compute ski-slope curves for the operator chains in the seven priority
models, both unfused (Orojenesis) and fused (DeFiNES-style, three overlap-storing modes),
and mark where 2 MiB/core and 1 MiB/core fall on each. Three outcomes are all actionable:
if 2 MiB is well past the knee, LX pressure is not the problem and Q2's layout hypothesis
is favoured; if it sits on the steep part, spilling and residency policy with a latency
objective (XLA MSA's benefit metric, computable from the existing per-op model) becomes the
priority; if fused chains move the knee substantially, that is a quantitative argument for
the KTIR fusion investment independent of any attention result. Requires no hardware,
which makes it the cheapest question on the list and a reasonable one to do first.

### Q7. How should the bucket and pad set be chosen under a serving SLO? *(new)*

**Why open.** Static shapes are mandatory on Spyre, vllm-spyre/sendnn-inference defines the
buckets by hand (`SENDNN_INFERENCE_WARMUP_PROMPT_LENS` × `_WARMUP_BATCH_SIZES`), requests
outside the compiled set are rejected, and no cited system chooses the set with a cost
model. Production sets the bounds — AWS Neuron auto-buckets at powers of two and dispatches
to the smallest fit under a ~2 GB per-NeuronCore artifact budget; vLLM's default capture
set is 51 sizes — but neither optimises. DietCode is the closest formulation and it is
learned, multiplicative (`f_MK · f_OCC · f_pad`), and its occupancy factor
`ceil_by(P/M, NumCores)` is exactly the 32-core ragged edge. This is arguably more
immediately fundable than Q4 because it changes a number a production deployment already
sets by hand.
**Related work.** DietCode, Nimble (residue dispatch; kernel-count-versus-performance
curve), Relax (symbolic upper-bound pre-allocation: 22% prefill / 40% decode activation
memory), SoD² (1.05× vs 1.16× of optimal peak), AWS Neuron bucketing, vLLM capture sets,
vllm-spyre configuration guide and RFC #9652, LENS (latency as a step function).
**Experiment.** (a) Reformulate the cost model's output as a *bucket-set score*:
`Σ_s p(s)·Cost(bucket(s)) + λ·CompileCost(|buckets|)` under an artifact-memory bound, with
p(s) taken from a real or synthetic arrival distribution. (b) Verify the multiplicative
form on Spyre: fit `f_pad` and `f_OCC` separately and test whether an additive padding term
is measurably worse, and whether `f_OCC` really tracks `ceil_by(P/M, 32)`. (c) Test
DietCode's local-padding claim directly — compare a predicated ragged tile against a
padded-fetch/sliced-writeback tile on the same shape, since the 17× figure is a T4 result
and an explicitly-staged scratchpad should behave differently. (d) Report the achieved
TTFT/TPOT against the hand-chosen bucket set on at least one priority model. Deliverable:
the first cost-model-selected bucket set for a static-shape inference accelerator.

### Q8. Does block-scaled FP8 survive stick alignment? *(new)*

**Why open.** FP8 is on the roadmap and the literature coverage of the interaction is
zero. The collision is concrete. MX fixes block size 32 with a shared E8M0 scale and
requires the last dimension divisible by 32 and the product of the others divisible by 32.
Row-wise (1×32) and column-wise (32×1) quantizations "cannot derive one from the other" —
a block-scaled tensor has **no free transpose**, which breaks any layout pass that reorders
axes. Scale tensors need their own swizzle (128×4 slices, interleave 0, 32, 64, 96, 1, 33,
…), and DeepGEMM requires the LHS scale to be TMA-aligned and transposed. Meanwhile a
128-byte stick is 2 MX blocks at fp16 and 8 at MXFP4, whose scales occupy 1/16 of a stick —
so the layout solver must place a companion tensor that is stick-sub-granular, cannot be
independently transposed, and quantises tile sizes along the reduction axis. Ladder is the
only system treating this as general compilation, and it targets GPU transaction widths.
And the format choice is not settled: at 8 bits the measured crest-factor evidence favours
MXINT8 over MXFP8 (crossover 7.55 versus measured Q3 = 2.96) at 0.63× energy and 0.79×
area.
**Related work.** OCP MX v1.0, TransformerEngine MXFP8, NVFP4, DeepGEMM, MARLIN (scale
repacking is what reaches the roofline; `b_opt ≈ 50`), QServe (20–90% dequant overhead),
Ladder `tType`, torchao `Layout`, DeepSeek-V3 ISCA'25 (partial sums moving between matrix
and scalar units), INT-vs-FP study.
**Experiment.** (a) Express the MX scale tensor as a Spyre layout and check whether the
existing solver can place it — specifically whether a sub-stick-granular companion tensor
with a fixed relationship to its data tensor is representable at all. (b) Enumerate which
current passes assume a free transpose or a reorderable axis and would therefore be
unsound under block scaling. (c) Decide where scale application belongs — inside the
DeepTools-generated inner loop, or as a separate vector-array pass — and price both with
the cost model; DeepSeek's dataflow argument says this is the first-order question.
(d) Determine whether bucket boundaries must be multiples of 32, since a boundary that is
not forces re-quantization rather than padding, which couples Q8 to Q7. This is a clean
layout-algebra question with a concrete deliverable and no hardware dependency for parts
(a)–(b).

### Q9. What is the rank-quality / compile-time Pareto front? *(new)*

**Why open.** Every system cited here reports rank quality *or* compile time, never the
front between them, and the roadmap gives a hard budget ("a few minutes per priority
model"). The relevant numbers exist but are scattered and incomparable: Roller emits top-10
at 13.3 s (V100) and 7.69 s (MI50); Welder compiles BERT in 244 s over 651 trials against
Ansor's 15,285 s over 8,000; the transform dialect costs ≤2.6% over the default pipeline;
onnx-mlir's BERT-base compile is 174 s (48% in LLVM opt), cut to 139 s by externalising
constants; DeFiNES spends 18 h for a 108-point sweep; AKG hand-tuned isl options to stay
under a minute for 99% of cases; TelaMalloc's entire headline is compile time. And
sendnn-inference already times warmup and ships a compilation cache precisely because the
budget binds in production.
**Related work.** Roller, Welder, Hidet, TelaMalloc, transform dialect (CGO'25), onnx-mlir
compile breakdown, AKG, DeFiNES, Autocomp schedule reuse (5.0× at ~10% of budget),
IISWC'22 warm-start (3.3–7.3× faster convergence), triton-dejavu,
`get_estimate_runtime_cache`.
**Experiment.** Sweep the candidate budget k (1, 2, 5, 10, 20) and the search formulation
(greedy / aligned construction / CP-SAT) from Q3, and plot achieved measured latency
against total compile-plus-search time per bucket, with the roadmap budget drawn as a
vertical line. Include the warm-start and persisted-cache variants as separate points,
since Autocomp and IISWC both claim most of the win at ~10% of the budget. Deliverable: a
front the roadmap can actually be planned against, and a defensible answer to "how many
candidates should we measure per bucket" that is currently chosen by intuition.

---

## 13. Cross-cutting synthesis: six claims worth holding

1. **Rank, don't predict.** TenSet (R² = −1818 winning on latency), TileLoom (17% error
   sufficient), tritonBLAS (94.7% selection efficiency), TLP (λ-rank beats MSE by 0.007 on
   top-1 yet the system wins 9.1×), Roller ("only needs to be accurate when tile shapes are
   fully aligned"), Prajapati (RMSE >100% globally, <10% near the optimum). Absolute
   accuracy is not the product — accuracy *near the optimum* is, and it is separately
   testable.
2. **Constrain the space before you model it.** Roller's alignment, Hidet's
   hardware-centric factors, Bolt's guidelines, Heron's constraints-by-construction,
   Ladder's alignment hints, CoSA's and Stream's capacity constraints. Every system that
   works on immature hardware shrinks the space first.
3. **Capacity is a filter almost everywhere; pricing it is the real gap.** The draft said
   nobody co-optimises tiling and allocation; that was wrong. Welder, Stream, DeFiNES,
   TileFlow, Elk, Baskaran and AKG all do. But Welder assigns infinite penalty above
   capacity, DeFiNES picks the lowest level that fits, Stream writes a hard inequality,
   PPCG refuses promotion, MLIR recurses inward — none models the cost *gradient* near
   capacity, and none feeds allocator outcome back into tile selection. Elk is the closest
   and it does not fuse.
4. **Fusion is not monotonically good.** FFM, FuseFlow, Pin-or-Fuse and DeFiNES
   independently reach this; the optimal fusion set is capacity- and shape-dependent, and
   DeFiNES shows the optimum is always interior. "Discover FlashAttention" is the wrong
   goal statement; "find the capacity-constrained optimal fusion set" is the right one.
5. **Where the backend is opaque, anchor empirically — and one tuned scalar is an
   accepted engineering point.** LENS (493% → 2.15% with two measured anchors per bucket),
   Roller's top-10, TileLoom's top-k profiling, Welder's `d.Profile` on the connect level,
   DOSA's learned residual on an analytical core, and XLA's shipped
   `compute + memory − min(·)·0.95`. Do not derive what you can measure once and compose;
   and do not assume `max(compute, memory)` because a doc page says so.
6. **The measured artefact is not the kernel.** nn-Meter: predict at the granularity the
   backend actually emits (SuperDSC, not ATen op). Habitat: per-op and importance-weighted
   end-to-end error differ by 2–3×. Mind Mappings: with heavy-tailed residuals, choose
   Huber over MSE. vllm-spyre: the served unit is a prefill graph plus a decode graph per
   bucket, under a compile budget — so the ultimate consumer of the model is a bucket set,
   not a loop nest.

---

## 14. Deliberately out of scope, and why

The critique named eight missing modalities. Three are now covered — polyhedral
compilation (§7.5), quantization (§10), dynamic shapes and bucketing (§9) — and compile
time, which the draft treated only as a constraint, is now an explicit axis (§9.3, Q9).
The remaining four are excluded on purpose, and the reasons are stated so a reader can
disagree with them.

**Sparsity.** Excluded. Spyre has no gather hardware exposed in the current IR and the
1H-2026 roadmap lists seven dense priority models with no sparse workload. The thread the
draft left dangling — Flashlight losing to FlexAttention on `block_mask` variants because
it has no sparsity optimisation — is real and is a genuine design question for a chip that
cannot skip blocks cheaply, but answering it requires hardware facts (does the DMA engine
support strided/indexed descriptors?) that are not in scope for a literature review. The
entry points when it becomes relevant are Sparseloop (MICRO'22, already cited in §8.3 for
its decoupling methodology), SparseTIR (ASPLOS'23), N:M structured sparsity, and the
block-sparse attention economics implicit in MegaBlocks. Triton's own related-work page
flags that polyhedral methods have "still to be successfully applied to sparse — or even
structured-sparse — neural networks," so the compiler-side story is thin in any case.

**Distributed and multi-chip.** Excluded, per the 1H-2026 roadmap, which puts multi-card
performance out of scope. This is why §5's Cursor/AsyncTP material reads as orphaned — it
is, deliberately, and it is retained only as an example of work division applied to
communication. Note that Inductor's *one* user-supplied-cost-model hook
(`config.estimate_op_runtime`, §3.3) is wired exclusively to communication-overlap passes,
so the day multi-card matters, the upstream integration story inverts: the hook that is
useless today becomes the natural one. Entry points when that happens: Alpa, GSPMD/Shardy,
PartIR, nnScaler, and the comm/compute-overlap compilers CoCoNet (ASPLOS'22), T3
(ASPLOS'24), Centauri (ASPLOS'24), Flux and TileLink. Also note the supported-features
matrix rules out prefill-decode disaggregation, which removes one common multi-device
lever from the serving objective in §9.

**Numerics and accuracy-aware compilation.** Mostly excluded, with one exception promoted
into Q4. On an fp16-native chip, online-softmax rescaling, accumulate width and FA4's
numerically-motivated conditional rescaling are compiler concerns, and DeepSeek's
FP22-accumulate observation shows accumulate width is a first-order architectural fact.
Q4 now carries the one question that gates a concrete deliverable — does the online-softmax
recurrence hold at the sequence lengths in the seven priority models. Beyond that,
accuracy-aware compilation, DL-compiler bug studies and fuzzing (NNSmith, Tzer) are out of
scope because torch-spyre's correctness story is currently an op-level numerical test
suite and the marginal research value is low relative to Q1–Q9. The one concrete borrowing
is onnx-mlir's `TEST_INSTRUCTION` (§6.2): assert *which* accelerator primitive was emitted,
not only that the numbers match. AutoMegaKernel's validator is the lone deadlock/race
analogue and is cited in §5.

**Energy and EDP as an objective.** Excluded, and this is the exclusion most worth
arguing with. FFM, Timeloop, ZigZag, Stream, DeFiNES and Interstellar all optimise energy
or EDP, and this review silently converts everything to latency. The justification is that
Spyre is a fixed 75 W part in an inference-serving deployment whose SLO is stated in TTFT
and TPOT, so latency is the contracted objective and power is a design-time constant rather
than a compile-time variable. The counter-argument is real: DeFiNES measures a 26× energy
span across tile size and mode, ZigZag 4.7× across temporal schedules for a single layer,
and the INT-vs-FP study argues format choice moves energy by 0.63× at matched throughput —
so if per-rack power ever becomes the binding constraint, the whole of §7.4 is already
written in the right currency and the objective can be swapped without changing the
formulation. Recorded here so the choice is visible rather than silent.

---

## 15. What I could not verify, and where sources conflict

These should be re-checked before any of them appears in a paper.

**Corrections carried in from the critique, recorded so they are not reintroduced.**
Roller is not an OSDI'22 best paper (the 2022 Jay Lepreau awards went to MemLiner, XRP and
Sieve — [UCLA](https://www.cs.ucla.edu/ucla-systems-group-won-jay-lepreau-best-paper-award-at-osdi/),
[Columbia](https://www.ee.columbia.edu/news/prof-asaf-cidon-and-team-receive-jay-lepreau-best-paper-award-16th-usenix-symposium-operating)).
Buffets is **ASPLOS 2019**, pp. 137–151, not ISCA 2019. Diesel is a **MAPL'18** 6-page
workshop paper co-located with PLDI'18 (DOI 10.1145/3211346.3211354), not a PLDI paper; its
full text is paywalled (ACM returned 403 on three attempts) and only the abstract, authors
and venue could be confirmed — **do not cite numbers or mechanism for Diesel**. DNNFusion's
"9.3×" is the maximum against the weakest baseline (PyTorch-Mobile), not an average.
MAESTRO has two non-interchangeable accuracy claims: "within 3.9% absolute error" (§4.5)
and "within 90–95% accuracy of actual open-source RTL" (intro). Buffets' real numbers are
2% control overhead over an 8 KB RAM and 1.53× / 5.39× EDP over double-buffered DMA and
caches; a web summariser produced "3–5%" and "2–3×", so any figure sourced from snippets
should be re-checked. Habitat's 11.8% is **end-to-end iteration** error; its per-operation
errors are 18.0% and 29.8%. XLA discussion #6407 writes the fusion priority sign-inverted
relative to shipped code; the code is authoritative.

**Moved here from the draft body.** (a) Axe's Trainium/NKI result — "matches hand-written
NKI on GEMM and beats it up to 1.44× (avg 1.26×) on MHA, in 228 lines against 1188" — could
not be confirmed; the abstract for 2601.19092 describes GPUs and device meshes. Confirm
from the PDF body before citing. (b) PyTorch release facts: that 2.12 shipped 2026-05-19
adding `torch.accelerator.Graph` for CUDA/XPU/PrivateUse1, that 2.10 deprecated TorchScript,
and that 2.11 added differentiable collectives and FlashAttention-4. None was verified in
this pass, and the 2.12 date sits oddly against the 2.11.0 tree that was actually read.

**Verified only from abstracts, metadata or secondary records.** T10 (SOSP'24) — the arXiv
abstract page and ACM/dblp metadata only; the full PDF was not read, so its internal
cost-model details are unconfirmed. Ladder's OSDI'24 numbers — the primary PDF returned
HTTP 403 from usenix.org; the figures come from the USENIX/MSR abstract and secondary
summaries and should be re-verified. Heron (ASPLOS'23) — ACM DL returns 403; abstract and
both speedup figures come from the Semantic Scholar record for DOI 10.1145/3582016.3582061,
and the evaluation section was not read. Graphene (ASPLOS'23) — ACM DL 403; abstract read
from the author's page, with no per-benchmark speedups published there. DNNDaSher — IEEE
Xplore and CSDL are paywalled; title, authors, DOI, abstract and numbers come from IBM
Research's own publication page. MLIR-AIR — the arXiv version (2510.14871) was read in
full, but search results indicating an ACM TRETS DOI (10.1145/3785670) could not be
confirmed (ACM returned 403), so the venue is unverified. IRON (FCCM'25) — architectural
parameters are not in the abstract. The MTIA ISCA 2025 paper, the Groq ISCA 2020/2022
papers and the SN40L Hot Chips deck all failed to parse; figures attributed to them come
from abstracts or vendor blogs. TelaMalloc's algorithmic details come from a third-party
summary (ACM PDF 403); MiniMalloc's head-to-head numbers against TelaMalloc are
unconfirmed. "Scratchpad Memory Management for Deep Learning Accelerators" (ICPP 2024,
10.1145/3673038.3673115) could not be retrieved and may contain a spill-policy comparison
worth checking. The DOSA text read was an arXiv HTML mirror at 2509.10702v1 whose numbering
does not match the MICRO'23 publication date; content matched, but confirm the canonical ID.

**No performance numbers published (so do not imply any).** IREE's data-tiling walkthrough;
iree-amd-aie; Cambricon triton-linalg; the Linalg/structured-ops paper (abstract reports
only "preliminary experimental results"); triton-dejavu.

**Named from memory, not fetched.** Ring Attention and Blockwise Parallel Transformers
(added to Q4 on the critique's recommendation) — author list and arXiv IDs were not
verified in this pass and must be fetched before citation. Likewise the four correctives to
§5's NVIDIA skew named by the critique: KernelGenBench
(<https://arxiv.org/abs/2607.27231>), BackendBench
(<https://github.com/meta-pytorch/BackendBench>), AMD's GEAK, and Sakana's CUDA Engineer
and Kevin-32B. Their claims are the critique's, not independently checked here.

**Direct conflicts.** The [Helion blog](https://pytorch.org/blog/helion/) claims Helion is
2.12–2.63× faster than TileLang on Mamba-2 chunk scan (H100); the
[TileLang paper](https://arxiv.org/abs/2504.17577) claims 1.77–2.10× over Triton on
linear-attention chunk scan/state (H100). Neither is a neutral third-party benchmark and I
could not reconcile them. Similarly, SonicMoE's numbers versus DeepGEMM differ between the
[arXiv abstract](https://arxiv.org/abs/2512.14080) (25% fwd / 15% bwd, Blackwell) and the
[Dao Lab blog](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/) (54% fwd / 35% bwd TFLOPS
on B300); treat the paper numbers as citable. And XLA's own documentation states the model
as `max(compute, memory) + launch overhead` while the shipped code implements
`compute + memory − min(·)·0.95`.

**Numbers read from HTML renders or search results rather than PDF tables.** Flashlight's
1.48× / ≥5× / 6–9% figures; TileSight's 12.35% MAPE and 99.66% retention; Nautilus's
256-measurement budget and timings; FlashAttention-4's 1.1–1.3× vs cuDNN and 2.1–2.7× vs
Triton (the [Tri Dao page](https://tridao.me/blog/2026/flash4/) was not fetched directly —
corroborated only by the Modal and PyTorch posts).

**Venue status.** TileLang is listed only as arXiv (April 2025) — not confirmed at any
venue. Hexcute and the Halide equality-saturation paper are confirmed CGO 2026.

**Thin or unreplicated.** The [CUDA Tile evaluation](https://arxiv.org/html/2604.23466v2)
is a third-party preprint that self-declares seven limitations including single-GPU
sampling and no Nsight roofline analysis; its 2.5×-over-FA2 result on B200 is
unreplicated. [AutoMegaKernel](https://arxiv.org/abs/2606.09682) is independent and not
peer reviewed. [Retire the Abstractions](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
contains zero measurements. [CUDA-L2](https://arxiv.org/abs/2512.02551) does not state
which GPU was used. FastKernels (arXiv 2605.23215) and [KForge](https://arxiv.org/abs/2511.13274)
resisted text extraction; their headline win-rates against production libraries could not
be recovered. Twill hand-compiles its output to CUDA because Triton lowers some decisions
incorrectly, so the split between search quality and lowering quality in its 1–2% gap is
unclear.

**Inferences, not sourced claims.** That Spyre's DeepTools is the same lineage as IBM's
2019 RaPiD DeepTools (author overlap makes it likely; no public document states it). The
MTIA-vs-Spyre SRAM-per-bandwidth ratio (my arithmetic from two vendor blogs). That
`propagate_layouts.py` + `insert_restickify.py` address the same problem onnx-mlir's ZHigh
passes and DNNDaSher address (structurally they do; the overlap in *coverage* is untested).
The stick-versus-MX-block arithmetic in §10 (2 MX blocks per fp16 stick, 8 at MXFP4, scales
at 1/16 of a stick) is my arithmetic from the MX spec and zDNN constants, not a cited claim.
That DeepTools double-buffers LX tiles — the Buffets-derived "effective capacity is 1 MiB"
argument in §7.1 is *conditional* on that and it is unverified; it is the first thing Q2
should establish.

**Unanswered mechanism questions.** No source explains how a layout algebra should model
a DMA/stick descriptor as the atom rather than a thread–value map. No public source
describes how DeepTools schedules compute/HBM overlap inside a single SuperDSC matmul.
FFM's pruning theorem assumes objectives and reservations are separable per-Einsum and
that joining is monotone; whether that survives a non-additive overlap term decided
outside the mapping is unresolved. Welder's inter-layer independence assumption is
similarly unverified for a per-core-partitioned scratchpad with unequal tail cores — see
Q3. And onnx-mlir's NNPA cost model reports r² per op but no rank-quality metric, so its
residuals bound its *fit*, not its *decision quality*.

---

## 16. Source index

Grouped by the section where each is load-bearing; sources used in more than one section
appear once, at their canonical home.

### §1 Tile abstractions and layout algebras

- Halide (CACM) — paper, 2018 — <https://andrew.adams.pub/halide_cacm.pdf>
- Triton (MAPL@PLDI) — paper, 2019 — <https://dl.acm.org/doi/abs/10.1145/3315508.3329973>
- Ansor (OSDI'20) — paper, 2020 — <https://arxiv.org/abs/2006.06762>
- TileLang — paper (arXiv), 2025 — <https://arxiv.org/abs/2504.17577>
- Linear Layouts (F₂) — paper, 2025–26 — <https://arxiv.org/abs/2505.23819>
- Axe — paper, 2026 — <https://arxiv.org/pdf/2601.19092>
- Hexcute (CGO 2026) — paper, 2025–26 — <https://arxiv.org/html/2504.16214>
- Graphene (ASPLOS 2023, NVIDIA) — paper, 2023 —
  <https://dl.acm.org/doi/10.1145/3582016.3582018>
- CuTe DSL / CUTLASS 4.0 — docs, 2025–26 —
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html>
- Gluon overview — docs, 2025–26 — <https://triton-lang.org/main/gluon/index.html>
- Helion — blog (PyTorch), 2025 — <https://pytorch.org/blog/helion/>
- CUDA Tile IR backend for Triton — blog (NVIDIA), 2025–26 —
  <https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/>
- Focus on Your Algorithm (CUDA Tile) — blog (NVIDIA), 2025 —
  <https://developer.nvidia.com/blog/focus-on-your-algorithm-nvidia-cuda-tile-handles-the-hardware/>
- Evaluating CUDA Tile on Hopper/Blackwell — preprint, 2026 —
  <https://arxiv.org/html/2604.23466v2>
- ThunderKittens (ICLR 2025) — paper, 2024–25 — <https://arxiv.org/abs/2410.20399>
- Pushing Tensor Accelerators Beyond MatMul (CGO 2026) — paper, 2025–26 —
  <https://arxiv.org/abs/2512.02371>
- Exo 2 (ASPLOS 2025) — paper, 2024–25 — <https://arxiv.org/pdf/2411.07211>
- Dato — paper, 2025 — <https://arxiv.org/abs/2509.06794>
- Pallas TPU details — docs (JAX), 2025–26 —
  <https://docs.jax.dev/en/latest/pallas/tpu/details.html>
- tilelang-ascend — repo, 2025–26 — <https://github.com/tile-ai/tilelang-ascend>

### §2 MLIR ecosystem and the tile-IR decision

- MLIR-AIR: From Loop Nests to Silicon — paper (arXiv), 2025 —
  <https://arxiv.org/pdf/2510.14871>
- The MLIR Transform Dialect (CGO 2025) — paper, 2025 —
  <https://www.steuwer.info/files/publications/2025/CGO-The-MLIR-Transform-Dialect.pdf>
- Transform Dialect reference — docs, 2026 — <https://mlir.llvm.org/docs/Dialects/Transform/>
- Composable and Modular Code Generation in MLIR (Linalg) — paper, 2022 —
  <https://arxiv.org/abs/2202.03293>
- IREE `stream` dialect — docs, 2026 — <https://iree.dev/reference/mlir-dialects/Stream/>
- IREE data-tiling walkthrough — blog, 2025 —
  <https://iree.dev/community/blog/2025-08-25-data-tiling-walkthrough/>
- IREE design roadmap — docs, 2026 — <https://iree.dev/developers/design-docs/design-roadmap/>
- IREE codegen configuration / tuning — docs, 2026 — <https://iree.dev/reference/tuning/>
- iree-amd-aie — repo, 2026 — <https://github.com/nod-ai/iree-amd-aie>
- torch-mlir roadmap — repo/docs, 2026 —
  <https://github.com/llvm/torch-mlir/blob/main/docs/roadmap.md>
- triton-shared — repo, 2026 — <https://github.com/microsoft/triton-shared>
- Cambricon triton-linalg — repo, 2026 — <https://github.com/Cambricon/triton-linalg>
- IRON (FCCM 2025) — paper, 2025 — <https://arxiv.org/abs/2504.18430>
- MLIR-AIE programming guide (IRON/ObjectFifo) — docs, 2026 —
  <https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/README.md>
- Versal AIE-ML Architecture Manual (AM020) — vendor docs, 2026 —
  <https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Tile-Architecture>
- AMD NPU — Linux kernel accel documentation — docs, 2026 —
  <https://docs.kernel.org/accel/amdxdna/amdnpu.html>
- Pallas Design (Mosaic) — docs (JAX), 2026 —
  <https://docs.jax.dev/en/latest/pallas/design/design.html>

### §3 torch.compile / TorchInductor, and fusion cost models

- State of torch.compile for training — blog, Aug 2025 —
  <https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/>
- `choices.py` (`score_fusion`, 2.11.0) — repo —
  <https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/choices.py>
- `scheduler.py` (`_get_estimated_runtime`, `speedup_by_fusion`, 2.11.0) — repo —
  <https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/scheduler.py>
- `comms.py` + `config.py` (`estimate_op_runtime`, 2.11.0) — repo —
  <https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/comms.py>
- `analysis/device_info.py` (2.11.0) — repo —
  <https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/analysis/device_info.py>
- `choices.py` (main) — repo —
  <https://github.com/pytorch/pytorch/blob/main/torch/_inductor/choices.py>
- `scheduler.py` (main) — repo —
  <https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py>
- `memory.py` (`reorder_for_peak_memory`) — repo —
  <https://github.com/pytorch/pytorch/blob/main/torch/_inductor/memory.py>
- `codegen/memory_planning.py` — repo —
  <https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/memory_planning.py>
- `autoheuristic/` artifacts — repo —
  <https://github.com/pytorch/pytorch/tree/main/torch/_inductor/autoheuristic>
- `tiling_utils.py` (`analyze_memory_coalescing`) — repo —
  <https://github.com/pytorch/pytorch/blob/main/torch/_inductor/tiling_utils.py>
- Inductor fusion escape hatch (open) — issue, 2025 —
  <https://github.com/pytorch/pytorch/issues/149603>
- Disabling codegen-specific fusions — dev-discuss, 2025 —
  <https://dev-discuss.pytorch.org/t/disabling-codegen-specific-fusions-in-torchinductor-for-per-op-kernel-generation/3226>
- Extend TorchInductor to more backends — issue, 2023–25 —
  <https://github.com/pytorch/pytorch/issues/99419>
- Why Is PyTorch Compile So Fast: Kernel Fusion — blog, 2026 —
  <https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/>
- PyTorch 2.12 release — blog, 2026 — <https://pytorch.org/blog/pytorch-2-12-release-blog/>
- PyTorch 2.9 release — blog, 2025 — <https://pytorch.org/blog/pytorch-2-9/>
- Intro to torch.compile with vLLM — blog, 2025 —
  <https://blog.vllm.ai/2025/08/20/torch-compile.html>
- vLLM torch.compile design — docs, 2026 —
  <https://docs.vllm.ai/en/latest/design/torch_compile/>
- vLLM fusion passes — docs, 2026 — <https://docs.vllm.ai/en/stable/design/fusions/>
- Unwrap custom ops and improve fusion — issue, 2025 —
  <https://github.com/vllm-project/vllm/issues/24629>
- XLA:GPU Priority-based fusion RFC — discussion, 2023 —
  <https://github.com/openxla/xla/discussions/6407>
- `priority_fusion.cc` (openxla/xla, main) — source, 2026 —
  <https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/priority_fusion.cc>
- `gpu_performance_model_base.{h,cc}` — source, 2026 —
  <https://github.com/openxla/xla/blob/main/xla/service/gpu/model/gpu_performance_model_base.h>
- Cost Models in XLA GPU – Present and Future — discussion, 2024 —
  <https://github.com/openxla/xla/discussions/10065>
- Operator Fusion in XLA: Analysis and Evaluation — paper, 2023 —
  <https://arxiv.org/abs/2301.13062>
- DNNFusion (PLDI'21) — paper, 2021 — <https://arxiv.org/abs/2108.13342>
- AStitch (ASPLOS'22) — paper, 2022 —
  <https://jamesthez.github.io/files/astitch-asplos22.pdf>
- Apollo (MLSys'22) — paper, 2022 —
  <https://proceedings.mlsys.org/paper_files/paper/2022/file/e175e8a86d28d935be4f43719651f86d-Paper.pdf>
- Chimera (HPCA'23) — paper, 2023 — <https://sizezheng.github.io/files/7A-3.pdf>

### §4 Automatic FlashAttention

- Flashlight — paper, 2025–26 — <https://arxiv.org/abs/2511.02043>
- Flashlight MLSys 2026 artifact/appendix — paper, 2026 —
  <https://mlsys.org/virtual/2026/poster/3540>
- pytorch-flashlight artifact — repo, 2026 — <https://github.com/bozhiyou/pytorch-flashlight>
- Neptune — paper, 2025 — <https://arxiv.org/abs/2510.08726>
- FlexAttention — paper, 2024 — <https://arxiv.org/abs/2412.05496>
- FlexAttention + FlashAttention-4 — blog (PyTorch), 2026 —
  <https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/>
- Mirage — paper, 2024–25 — <https://arxiv.org/abs/2405.05751>
- Nautilus — preprint, 2026 — <https://arxiv.org/abs/2604.14825> ·
  <https://arxiv.org/html/2604.14825v1>
- FFM (Fast and Fusiest) — paper, 2026 — <https://arxiv.org/abs/2602.15166>
- FlatAttention — paper, 2026 — <https://arxiv.org/abs/2604.02110>
- SDPA on streaming dataflow — paper, 2024 — <https://arxiv.org/abs/2404.16629>
- FuseFlow — paper, 2026 — <https://arxiv.org/abs/2511.04768>
- Event Tensor — paper, 2026 — <https://arxiv.org/abs/2604.13327>
- Reverse engineering FlashAttention-4 — blog (Modal), 2025 —
  <https://modal.com/blog/reverse-engineer-flash-attention-4>
- FlashAttention-4 — blog (Tri Dao), 2026 — <https://tridao.me/blog/2026/flash4/>
- FlashAttention-3 — paper, 2024 — <https://arxiv.org/abs/2407.08608>

### §5 Hand-written kernels

- FlashMLA seesaw deep dive — tech report, 2025 —
  <https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md>
- FlashMLA Hopper FP8 sparse decoding — tech report, 2025 —
  <https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md>
- QuACK: memory-bound kernels at speed of light — blog, 2025 —
  <https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md>
- SonicMoE (ICLR 2026) — paper, 2025–26 — <https://arxiv.org/abs/2512.14080>
- SonicMoE on Blackwell — blog (Dao Lab), 2026 —
  <https://dao-lab.ai/blog/2026/sonicmoe-blackwell/>
- MegaBlocks (MLSys 2023) — paper, 2022 — <https://arxiv.org/abs/2211.15841>
- Triton persistent cache-aware grouped GEMM — blog (PyTorch), 2025 —
  <https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/>
- Cross-platform fused MoE dispatch in Triton — preprint, 2026 —
  <https://arxiv.org/html/2605.23911v1>
- Mixture-of-Kittens — blog (Cursor), 2026 — <https://cursor.com/blog/mixture-of-kittens>
- HipKittens — preprint, 2025 — <https://arxiv.org/abs/2511.08083>
- Look Ma, No Bubbles! (Llama-1B megakernel) — blog (Hazy Research), 2025 —
  <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>
- Warp Specialization in Triton: design and roadmap — blog (PyTorch), 2026 —
  <https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/>
- Twill — preprint, 2025 — <https://arxiv.org/abs/2512.18134>
- Gluon: Explicit Performance — blog, 2025 —
  <https://www.lei.chat/posts/gluon-explicit-performance/>
- KernelBench (ICML 2025) — paper, 2025 — <https://arxiv.org/abs/2502.10517>
- Surprisingly Fast AI-Generated Kernels — blog (Stanford CRFM), 2025 —
  <https://crfm.stanford.edu/2025/05/28/fast-kernels.html>
- CUDA-L2 — preprint, 2025 — <https://arxiv.org/abs/2512.02551>
- Automated kernel generation in the era of LLMs (survey) — preprint, 2026 —
  <https://arxiv.org/html/2601.15727v3>
- AutoMegaKernel — preprint, 2026 — <https://arxiv.org/abs/2606.09682>
- KForge — preprint, 2025 — <https://arxiv.org/abs/2511.13274>
- KernelGenBench (unverified) — preprint, 2026 — <https://arxiv.org/abs/2607.27231>
- BackendBench (unverified) — repo, 2026 — <https://github.com/meta-pytorch/BackendBench>
- Structured Mojo Kernels Part 1 — blog (Modular), 2026 —
  <https://www.modular.com/blog/structured-mojo-kernels-part-1-peak-performance-half-the-code>
- Retire the Abstractions — blog (Hazy Research), 2026 —
  <https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions>
- TT-Metalium Programming Guide — repo/docs, 2025 —
  <https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md>

### §6 Dataflow-accelerator compilers, and IBM's own precedent

- zDNN `zdnn_private.h` (stick constants) — repo, 2021–2026 —
  <https://github.com/IBM/zDNN/blob/main/zdnn/zdnn_private.h>
- zDNN README (zTensors, layouts, transform API) — repo, 2026 —
  <https://github.com/IBM/zDNN/blob/main/README.md>
- Compiling ONNX Neural Network Models Using MLIR — paper, 2020 —
  <https://arxiv.org/abs/2008.08272>
- ONNX-MLIR (Asia LLVM Dev Meeting, Tokyo) — slides, 2025 —
  <https://llvm.org/devmtg/2025-06/slides/technical-talk/le-onnx.pdf>
- onnx-mlir: adding a new custom accelerator — docs, 2026 —
  <https://github.com/onnx/onnx-mlir/blob/main/docs/AddCustomAccelerators.md>
- onnx-mlir `NNPACompilerOptions.cpp` — source, 2026 —
  <https://github.com/onnx/onnx-mlir/blob/main/src/Accelerators/NNPA/Compiler/NNPACompilerOptions.cpp>
- onnx-mlir `DevicePlacementHeuristic.cpp` — source, 2026 —
  <https://github.com/onnx/onnx-mlir/blob/main/src/Accelerators/NNPA/Conversion/ONNXToZHigh/DevicePlacementHeuristic.cpp>
- onnx-mlir `PerfModelArch15.inc` — source, 2025–26 —
  <https://github.com/onnx/onnx-mlir/blob/main/src/Accelerators/NNPA/Conversion/ONNXToZHigh/PerfModelArch15.inc>
- onnx-mlir `utils/NNPAOpPerfModel` (model generation) — source, 2025–26 —
  <https://github.com/onnx/onnx-mlir/tree/main/utils/NNPAOpPerfModel>
- onnx-mlir ZHigh layout-optimisation passes — source, 2022–26 —
  <https://github.com/onnx/onnx-mlir/tree/main/src/Accelerators/NNPA/Transform/ZHigh>
- onnx-mlir `RewriteONNXForZHigh.cpp` (SplitLargeMatMul) — source, 2026 —
  <https://github.com/onnx/onnx-mlir/blob/main/src/Accelerators/NNPA/Conversion/ONNXToZHigh/RewriteONNXForZHigh.cpp>
- onnx-mlir NNPA how-to-use and test — docs, 2026 —
  <https://onnx.ai/onnx-mlir/AccelNNPAHowToUseAndTest.html>
- onnx-mlir (project root) — repo, 2026 — <https://github.com/onnx/onnx-mlir>
- IBM Z Deep Learning Compiler (zDLC) — repo, 2022–26 — <https://github.com/IBM/zDLC>
- DNNDaSher (IEEE Micro 2024) — paper, 2024 —
  <https://ieeexplore.ieee.org/document/10596296/> ·
  <https://www.research.ibm.com/publications/dnndasher-a-compiler-framework-for-dataflow-compatible-end-to-end-acceleration-on-ibm-aiu>
- DeepTools (RaPiD) — paper, 2019 —
  <https://research.ibm.com/publications/deeptools-compiler-and-execution-runtime-extensions-for-rapid-ai-accelerator>
- PyTorch-native support for IBM Spyre — blog (IBM), 2026 —
  <https://research.ibm.com/blog/pytorch-support-ibm-spyre>
- Spyre PyTorch enabling status, 1H 2026 — dev-discuss, 2026 —
  <https://dev-discuss.pytorch.org/t/ibm-spyre-accelerator-pytorch-enabling-status-and-feature-plan-1h-2026/3319>
- Lifting the cover on the IBM Spyre Accelerator — blog (IBM), 2025 —
  <https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator>
- IBM/deepview — repo, 2026 — <https://github.com/IBM/deepview>
- ktir-mlir-frontend — repo, 2026 — <https://github.com/torch-spyre/ktir-mlir-frontend>
- TTNN Optimizer design spec — docs, 2026 —
  <https://docs.tenstorrent.com/tt-mlir/specs/ttnn-optimizer.html>
- tt-mlir dialect overview — docs, 2026 — <https://docs.tenstorrent.com/tt-mlir/overview.html>
- TileLoom — paper, 2025–26 — <https://arxiv.org/abs/2512.22168> ·
  <https://arxiv.org/html/2512.22168v2>
- SambaNova SN40L (MICRO 2024) — paper, 2024 — <https://arxiv.org/abs/2405.07518>
- Why Dataflow Matters More Than Ever — blog (SambaNova), 2025 —
  <https://sambanova.ai/blog/why-dataflow-matters-more-than-ever>
- Groq TSP multiprocessor (ISCA 2022) — paper, 2022 —
  <https://dl.acm.org/doi/10.1145/3470496.3527405>
- The Architecture of Groq's LPU — blog, 2024 —
  <https://blog.codingconfessions.com/p/groq-lpu-design>
- What is a Language Processing Unit? — blog (Groq), 2025 —
  <https://groq.com/blog/the-groq-lpu-explained>
- Supporting PyTorch on the Cerebras WSE — blog, ~2023 —
  <https://www.cerebras.ai/blog/supporting-pytorch-on-the-cerebras-wafer-scale-engine>
- Cerebras SDK (CSL) — docs, 2025 — <https://sdk.cerebras.ai/>
- MTIA v2 — blog (Meta), 2024 —
  <https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/>
- Meta's second-generation AI chip (ISCA 2025) — paper, 2025 —
  <https://dl.acm.org/doi/10.1145/3695053.3731409>
- Trainium2 architecture guide for NKI — docs, 2026 —
  <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html>
- Neuron Kernel Interface (NKI) — docs, 2026 —
  <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html>
- AccelOpt — paper, 2025 — <https://arxiv.org/abs/2511.15915>
- KernelEvolve — paper, 2025 — <https://arxiv.org/abs/2512.23236>
- LENS (NPU latency prediction) — paper, 2026 — <https://arxiv.org/html/2606.18042v2>
- Evaluating emerging accelerators: IPU, RDU, GPUs (ICPE 2024) — paper, 2024 —
  <https://arxiv.org/html/2311.04417v3>
- Mapping a model to an IPU system (Poplar) — docs, 2024 —
  <https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/map-model-to-ipu-system.html>
- NVIDIA vs AMD vs Tenstorrent architectural deep dive — blog (third party), 2026 —
  <https://blog.gpu.net/posts/2026/june/new-blog-june12/>

### §7 On-chip memory, tiling, and joint optimisation

- Buffets (ASPLOS 2019) — paper, 2019 —
  <https://ysshao.github.io/assets/papers/Buffet_ASPLOS19_Final.pdf>
- Welder (OSDI'23) — paper, 2023 — <https://www.usenix.org/system/files/osdi23-shi.pdf> ·
  <https://www.usenix.org/conference/osdi23/presentation/shi>
- Ladder (OSDI'24) — paper, 2024 —
  <https://www.usenix.org/system/files/osdi24-wang-lei.pdf> ·
  <https://www.usenix.org/conference/osdi24/presentation/wang-lei>
- Rammer (OSDI'20) — paper, 2020 — <https://www.usenix.org/system/files/osdi20-ma.pdf>
- ZigZag (IEEE TC 2021) — paper, 2021 — <https://arxiv.org/abs/2007.11360> ·
  <https://github.com/KULeuven-MICAS/zigzag>
- DeFiNES (HPCA'23) — paper, 2023 — <https://arxiv.org/abs/2212.05344>
- Stream (IEEE TC 2025) — paper, 2025 — <https://arxiv.org/abs/2212.10612> ·
  <https://github.com/KULeuven-MICAS/stream>
- TileFlow (MICRO'23) — paper, 2023 —
  <https://gulang2019.github.io/files/tileflow-micro23.pdf>
- T10 (SOSP'24) — paper, 2024 — <https://arxiv.org/abs/2408.04808>
- Elk (MICRO'25) — paper, 2025 — <https://arxiv.org/abs/2507.11506>
- AKG (PLDI 2021) — paper, 2021 — <https://doi.org/10.1145/3453483.3454106>
- Tensor Comprehensions — paper, 2018 — <https://arxiv.org/pdf/1802.04730>
- PPCG `ppcg_options.c` — source, ongoing —
  <https://raw.githubusercontent.com/Meinersbur/ppcg/master/ppcg_options.c>
- Baskaran et al. (PPoPP 2008) — paper, 2008 —
  <https://www.ece.lsu.edu/jxr/Publications-pdf/tr5-08.pdf>
- Tiramisu (CGO 2019) — paper, 2018–19 — <https://arxiv.org/pdf/1804.10694v3>
- MLIR AffineDataCopyGeneration — source, ongoing —
  <https://mlir.llvm.org/doxygen/AffineDataCopyGeneration_8cpp_source.html>
- Analytical time modeling and optimal tile size selection (PPoPP'17) — paper, 2017 —
  <https://www.pollylabs.org/publications/grosser-2017-Simple-Accurate-Analytical-Time-Modeling-and-Optimal-Tile-Size-Selection-for-GPGPU-Stencils.pdf>
- Triton "Related Work" (polyhedral vs scheduling languages) — docs, ongoing —
  <https://triton-lang.org/main/programming-guide/chapter-2/related-work.html>
- TensorComprehensions (archived 2023-04-28) — repo —
  <https://github.com/facebookresearch/TensorComprehensions>
- Diesel (MAPL'18 workshop) — paper, 2018 — <https://dl.acm.org/doi/10.1145/3211346.3211354>
- TelaMalloc (ASPLOS'23) — paper, 2023 — <https://dl.acm.org/doi/10.1145/3567955.3567961>
- MiniMalloc (ASPLOS'23) — paper, 2023 —
  <https://research.google/pubs/minimalloc-a-lightweight-memory-allocator-for-hardware-accelerated-machine-learning/>
- COSMA — paper, 2023 — <https://arxiv.org/abs/2311.18246>
- Timeloop — paper, 2019 — <https://accelergy.mit.edu/timeloop.pdf>
- CoSA (ISCA'21) — paper, 2021 — <https://arxiv.org/abs/2105.01898> ·
  <https://arxiv.org/pdf/2105.01898>
- Marvel (TACO'22) — paper, 2022 — <https://arxiv.org/abs/2002.07752>
- Interstellar (ASPLOS'20) — paper, 2020 — <https://arxiv.org/abs/1809.04070>
- Orojenesis (ISCA'24) — paper, 2024 —
  <https://people.csail.mit.edu/emer/media/papers/2024.06.isca.orojenesis.pdf>
- OpenXLA MemorySpaceAssignment — repo, 2026 —
  <https://github.com/openxla/xla/tree/main/xla/service/memory_space_assignment>
- Checkmate (MLSys'20) — paper, 2020 — <https://arxiv.org/abs/1910.02653>
- Moccasin (ICML'23) — paper, 2023 — <https://arxiv.org/abs/2304.14463>
- OnSRAM (TECS'22) — paper, 2022 — <https://dl.acm.org/doi/10.1145/3530909>
- Pin or Fuse? (CGO'23) — paper, 2023 — <https://dl.acm.org/doi/10.1145/3579990.3580017>
- tt-metal memory allocator tech report — repo, 2026 —
  <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/memory/allocator.md>
- TPU pipelining (Pallas) — docs, 2026 —
  <https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html>
- DOSA — paper, 2023 (arXiv HTML mirror) — <https://arxiv.org/html/2509.10702v1>

### §8 Tile-size search, mapping canon, ranking methodology

- Roller (OSDI'22) — paper, 2022 — <https://www.usenix.org/conference/osdi22/presentation/zhu> ·
  <https://www.microsoft.com/en-us/research/publication/roller-fast-and-efficient-tensor-compilation-for-deep-learning/>
- AutoTVM (NeurIPS 2018) — paper, 2018 — <https://arxiv.org/abs/1805.08166>
- Hidet (ASPLOS'23) — paper, 2023 — <https://arxiv.org/abs/2210.09603>
- Bolt (MLSys'22) — paper, 2022 —
  <https://proceedings.mlsys.org/paper_files/paper/2022/file/1f8053a67ec8e0b57455713cefdd8218-Paper.pdf>
- Heron (ASPLOS'23) — paper, 2023 — <https://dl.acm.org/doi/10.1145/3582016.3582061>
- MAESTRO (MICRO'19) — paper, 2019 — <https://arxiv.org/abs/1805.02566>
- Mind Mappings (ASPLOS'21) — paper, 2021 —
  <https://www.kartikhegde.net/media/Mind_Mappings_ASPLOS2021_CR.pdf>
- Sparseloop (MICRO'22) — paper, 2022 — <https://arxiv.org/abs/2205.05826>
- FAST (ASPLOS'22) — paper, 2022 — <https://arxiv.org/abs/2105.12842>
- nn-Meter (MobiSys'21) — paper, 2021 —
  <https://air.tsinghua.edu.cn/pdf/nn-Meter-Towards-Accurate-Latency-Prediction-of-Deep-Learning-Model-Inference-on-Diverse-Edge-Devices.pdf>
- Habitat (ATC'21) — paper, 2021 — <https://arxiv.org/abs/2102.00527>
- TLP (ASPLOS'23) — paper, 2023 — <https://arxiv.org/abs/2211.03578>
- TenSet (NeurIPS 2021 D&B) — paper, 2021 —
  <https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf>
- TpuGraphs (NeurIPS'23 D&B) — paper, 2023 — <https://arxiv.org/abs/2308.13490>
- A Learned Performance Model for TPUs (MLSys'21) — paper, 2021 —
  <https://arxiv.org/abs/2008.01040>
- tritonBLAS — paper (AMD), 2025 — <https://arxiv.org/abs/2512.04226>
- TileSight — preprint, 2026 — <https://arxiv.org/html/2607.22432v1>
- Pruner (ASPLOS'25) — paper, 2025 — <https://arxiv.org/abs/2402.02361>
- Demystifying Map Space Exploration for NPUs (IISWC'22) — paper, 2022 —
  <https://arxiv.org/abs/2210.03731>
- `triton.autotune` — docs, 2025 —
  <https://triton-lang.org/main/python-api/generated/triton.autotune.html>
- nvMatmulHeuristics (CUTLASS) — docs, 2025 —
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/heuristics.html>
- Autocomp — paper, 2025 — <https://arxiv.org/abs/2505.18574>
- triton-dejavu — repo (IBM), 2025 — <https://github.com/IBM/triton-dejavu>

### §9 Static shapes, bucketing, serving objective

- DietCode (MLSys'22) — paper, 2022 —
  <https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html>
- Nimble (MLSys'21) — paper, 2021 — <https://arxiv.org/abs/2006.03031>
- Relax (ASPLOS'25) — paper, 2025 — <https://arxiv.org/abs/2311.02103>
- SoD² (ASPLOS'24) — paper, 2024 — <https://arxiv.org/abs/2403.00176>
- NxD Inference bucketing (AWS Neuron) — docs, 2025–26 —
  <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html>
- vLLM CUDA Graphs design / CompilationConfig — docs, 2026 —
  <https://docs.vllm.ai/en/stable/design/cuda_graphs/>
- vllm-spyre / sendnn-inference — repo, 2025–26 —
  <https://github.com/vllm-project/vllm-spyre>
- vllm-spyre configuration guide — docs, 2026 —
  <https://docs.vllm.ai/projects/spyre/en/latest/user_guide/configuration.html>
- vllm-spyre supported features matrix — docs, 2026 —
  <https://docs.vllm.ai/projects/spyre/en/latest/user_guide/supported_features.html>
- [RFC] Add support for IBM Spyre accelerator — issue, 2024 —
  <https://github.com/vllm-project/vllm/issues/9652>

### §10 Quantization as a layout problem

- OCP Microscaling Formats (MX) Specification v1.0 — spec, 2023 —
  <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
- MXFP8 — Transformer Engine documentation — docs, 2026 —
  <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html>
- DeepGEMM — repo, 2025–26 — <https://github.com/deepseek-ai/DeepGEMM>
- MARLIN — paper/repo, 2024 — <https://arxiv.org/abs/2408.11743>
- QServe (MLSys'25) — paper, 2024–25 — <https://arxiv.org/abs/2405.04532>
- TorchAO — paper/repo, 2025 — <https://arxiv.org/abs/2507.16099>
- Insights into DeepSeek-V3 (ISCA 2025) — paper, 2025 — <https://arxiv.org/abs/2505.09343>
- INT vs FP: fine-grained low-bit quantization formats — paper, 2025 —
  <https://arxiv.org/abs/2510.25602>
