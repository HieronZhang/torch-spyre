# Completeness critique

## A. Errors and overstatements (checked)

**A1. "Roller (OSDI'22, best paper)" — almost certainly wrong.** OSDI '22 Jay Lepreau Best Paper Awards went to MemLiner, XRP, and Sieve/"Automatic Reliability Testing for Cluster Management Controllers". Roller carries no award badge. Drop "best paper" — the argument does not need it. ([UCLA](https://www.cs.ucla.edu/ucla-systems-group-won-jay-lepreau-best-paper-award-at-osdi/), [Columbia](https://www.ee.columbia.edu/news/prof-asaf-cidon-and-team-receive-jay-lepreau-best-paper-award-16th-usenix-symposium-operating))

**A2. §2 "[`_get_estimated_runtime`] is used only for comm/compute overlap reordering in `comms.py`, never for fusion or tiling" — false in the tree you cite.** In the installed 2.11.0, `scheduler.py:4148` calls `node2._get_estimated_runtime()` inside `speedup_by_fusion`'s multi-template path and feeds it to `_estimate_fused_epilogue_runtime` (`scheduler.py:2769`), which scales the epilogue estimate by an extra-bytes ratio and compares against unfused template timings. So Inductor *does* have one runtime-estimate-driven fusion decision — gated on `MultiTemplateBuffer` + Triton templates, which is why it never fires for you, but "there is no runtime model in the fusion path" is too strong as written.

**A3. §2/§8/Q5 miss the one upstream injection point.** `torch._inductor.config.estimate_op_runtime` (config.py:442) accepts a **user-supplied callable**, dispatched at `comms.py:2138–2146`. Also `torch/_inductor/analysis/device_info.py` is a datasheet TOPS/DRAM-bandwidth table keyed by device name — the natural place to register Spyre so the roofline stops returning 0. "A PrivateUse1 backend inherits no timing model at all… no upstream machinery will consume it unaided" needs to become "one hook exists, and it is wired only to comms reordering."

**A4. Q5's "two things to determine first" are already answered by the source you have.** The `benchmark_fusion` guard is literally CPU-specific, not a non-Triton proxy: `scheduler.py:4013-4014` reads `if device.type == "cpu" and config.cpu_backend != "triton"`. And `speedup_by_fusion` early-returns only when `not config.benchmark_fusion and not is_multi_template`. Delete those two sub-tasks.

**A5. §7.4 misses upstream's own measure-and-persist cache.** `config.runtime_estimations_mms_benchmark` + `get_estimate_runtime_cache()` (`scheduler.py:1336-1361`) benchmarks mm-like nodes and persists the result keyed by snode; `runtime_estimations_use_nccl_lib_estimations` uses vendor-library estimates as anchors. That is LENS's "measured anchors" pattern already inside Inductor, and a better citation than triton-dejavu alone for the warm-start recommendation.

**A6. Q5 is not open in the form stated.** XLA:GPU's **PriorityFusion** already ranks fusion candidates by predicted runtime deltas from `GpuPerformanceModel::EstimateRunTimes` — a production compiler that replaced a heuristic fusion score with an analytical latency model. Cite [openxla/xla#6407 (RFC)](https://github.com/openxla/xla/discussions/6407), [#10065 "Cost Models in XLA GPU"](https://github.com/openxla/xla/discussions/10065), and [Operator Fusion in XLA: Analysis and Evaluation](https://arxiv.org/abs/2301.13062). Narrow Q5 to "inside Inductor, for a non-GPU backend, where benchmarking is unaffordable."

**A7. §8.3 / Q1 "Nobody has published rank correlation of an analytical model against measured latency behind such a backend" — over-claimed.** [TpuGraphs (NeurIPS'23 D&B)](https://arxiv.org/abs/2308.13490) has a **tile-size configuration collection** scored by Kendall's τ against measured TPU latency behind the proprietary XLA:TPU backend, and [Kaufman et al., A Learned Performance Model for TPUs (MLSys'21)](https://arxiv.org/abs/2008.01040) reports ranking quality on the same setup. Both are learned, not analytical, and TPU is not an explicit-NoC dataflow chip — that is the surviving novelty, so say exactly that.

**A8. Axe's Trainium result is unverified and load-bearing.** The arXiv abstract for 2601.19092 ("Axe: A Simple Unified Layout Abstraction for Machine Learning Compilers", Hou, Jin, …, Tianqi Chen) describes GPUs and device meshes; it does not mention Trainium or NKI. Your §1.2 and §8.3 both lean on "the only layout algebra with published results on a non-GPU systolic accelerator … 1.44× over NKI, 228 vs 1188 lines". Confirm from the PDF body or move to §11.

**A9. §0 "The unit of compilation is now the tile, and that argument is over."** Overstated, and the review's own blind spot proves it: XLA, IREE, AKG and Mosaic all still compile from structured loop IR with compiler-chosen tiling, and MLIR's Linalg + `transform` dialect is a live competing answer to "who owns the schedule." The tile-as-value camp won *in GPU DSLs*; that is a narrower claim.

**A10. §6 "no published system spills within a fused kernel / nobody co-optimizes tiling and scratchpad allocation" — too strong.** See B3 below (Welder, AKG/PPCG memory promotion, Stream/DeFiNES).

**A11. Unverified PyTorch facts.** "2.12 shipped 2026-05-19", "2.11 added differentiable collectives and FlashAttention-4", "`torch.accelerator.Graph`" — none checked, and the 2.12 date sits oddly against a 2.11.0 tree. Move to §11 or verify.

---

## B. Missing systems (concrete, with URLs)

**B1. IBM's own directly-analogous compiler is absent.** onnx-mlir's NNPA backend for the IBM Z Integrated Accelerator lowers ONNX → **ZHigh/ZLow** with explicit `stickify`/`unstickify` ops over zDNN's stick layout, and its layout-conversion-minimization passes are the published ancestor of `insert_restickify`/`optimize_restickify` — the term "stick" comes from there. This is a bigger structural precedent than DNNDaSher and you cite neither.
- <https://github.com/onnx/onnx-mlir> · <https://onnx.ai/onnx-mlir/AccelNNPAHowToUseAndTest.html> · <https://github.com/IBM/zDLC> · "Compiling ONNX Neural Network Models Using MLIR" <https://arxiv.org/abs/2008.08272>

**B2. The production consumer of torch-spyre is absent.** [IBM/vllm-spyre](https://github.com/vllm-project/vllm-spyre) (docs: <https://vllm-spyre.readthedocs.io/>) defines the *objective function* your cost model is optimizing — warmup/compile budget, shape buckets, prefill/decode split. §8 discusses no serving-level objective at all.

**B3. Welder (OSDI'23) is the single largest omission.** "Welder: Scheduling Deep Learning Memory Access via Tile-graph" — holistic tile-level data-movement scheduling across memory layers, explicitly trading intra- vs inter-operator reuse with a **tile-traffic cost model**, on a multi-level hierarchy. It is §3's fusion tradeoff and §6's "joint tiling + on-chip residency" gap, already published and validated against measured latency. <https://www.usenix.org/conference/osdi23/presentation/shi>
Same lineage, also missing: **Ladder (OSDI'24)** (dtype/layout transformation — quantization × layout, exactly your FP8 blind spot) and **Rammer (OSDI'20)** (inter+intra-operator scheduling onto vDevices/rTasks — prior art for `work_division.py`/`core_mapping.py`).

**B4. The KU Leuven layer-fusion line.** Both do capacity-constrained fused scheduling for multi-core dataflow accelerators with explicit on-chip memory — i.e. FFM's thesis, three years earlier, with open tooling.
- **Stream**: <https://arxiv.org/abs/2212.10612> · <https://github.com/KULeuven-MICAS/stream>
- **DeFiNES** (HPCA'23, depth-first/layer-fused DSE with analytical memory modeling — produces exactly Q6's capacity-vs-traffic curves for *fused stacks*): <https://arxiv.org/abs/2212.05344>
- **ZigZag** (the underlying mapping DSE): <https://github.com/KULeuven-MICAS/zigzag>

**B5. Accelerator-mapping canon you skipped.** MAESTRO (MICRO'19, data-centric reuse cost model), **Buffets** (ISCA'19 — the canonical taxonomy for explicit-decoupled data orchestration in software-managed scratchpads; §5's framing needs it), **Mind Mappings** (ASPLOS'21 — gradient search on a differentiable surrogate; DOSA's direct ancestor), **Sparseloop** (MICRO'22), **FAST** (ASPLOS'22), **Heron** (ASPLOS'23, constraint-based schedule-space *construction* — belongs in §10 claim 2 alongside Roller/Hidet).

**B6. MLIR ecosystem — nearly absent, and decisive for the KTIR decision.** You cite only tt-mlir and CUDA Tile IR. Missing: Linalg + [`transform` dialect](https://mlir.llvm.org/docs/Dialects/Transform/) (schedules as IR), [IREE](https://iree.dev) (the main open PyTorch→custom-accelerator path, with its own tiling/fusion/bufferization and `stream` scheduling), [torch-mlir](https://github.com/llvm/torch-mlir), [triton-shared](https://github.com/microsoft/triton-shared) (Triton→Linalg — the route most non-GPU Triton backends actually take), and **MLIR-AIE / IRON** (<https://github.com/Xilinx/mlir-aie>) — AMD NPU: spatial tiles, explicit DMA, software-managed L1, no cache, i.e. the closest *open-source* analogue to Spyre's programming model. Also **Mosaic**, the compiler under Pallas/TPU that you cite only via docs, and **Graphene** (ASPLOS'23, NVIDIA — a tile/layout IR predating linear layouts).

**B7. Fusion cost models outside Inductor/XLA.** DNNFusion (PLDI'21), AStitch (ASPLOS'22), **Apollo** (MLSys'22 — partition-based fusion under memory constraints), Chimera (HPCA'23, fused GEMM chains). §2 and Q5 read as if only Inductor and vLLM exist.

**B8. Latency-prediction methodology.** nn-Meter (MobiSys'21, kernel-level latency prediction on real devices), Habitat (ATC'21, cross-device runtime prediction), TLP (ASPLOS'23, cost model for tensor-program tuning). §7.3 jumps from TenSet straight to your own scoring plan without the intervening methodology literature.

**B9. §4 LLM-kernel coverage is NVIDIA-skewed in exactly the way it complains about.** Add **GEAK** (AMD's agentic Triton kernel generator, non-NVIDIA), [**KernelGenBench**](https://arxiv.org/abs/2607.27231) (multi-source, **multi-chip**), [**BackendBench**](https://github.com/meta-pytorch/BackendBench) (Meta — evaluates *whole backends*: 271 ops correctness / 124 perf against OpInfo + HF-traced shapes; this is the obvious correctness harness for a PrivateUse1 backend and belongs in §8 too), plus Sakana's CUDA Engineer (the canonical reward-hacking case study you allude to) and Kevin-32B.

---

## C. Whole modalities missing

1. **Polyhedral compilation — entirely absent.** Most damaging: **AKG** (PLDI'21, polyhedral for Huawei Ascend NPUs) and PPCG-style **memory promotion**, which is literally "choose tile sizes and promote tiles into a software-managed scratchpad, jointly, as one affine problem." Q3 is framed as if this line does not exist. Also Tiramisu (CGO'19), Diesel (PLDI'18), Polygeist, MLIR `affine`.
2. **Quantization-aware compilation — zero coverage, and the roadmap says FP8.** Block-scaled formats (OCP MX, MXFP4/NVFP4) make the *scale tensor's* layout and block granularity a first-class layout-algebra problem that will collide with 128-byte stick alignment. Add Ladder (OSDI'24), Marlin, QServe/QoQ, torchao, TransformerEngine scaling recipes. DeepGEMM's per-block scaling appears in §4 but is never framed as compilation.
3. **Sparsity — absent as a topic.** Sparseloop (MICRO'22), SparseTIR (ASPLOS'23), N:M structured sparsity, block-sparse attention economics. You note Flashlight loses to FlexAttention on `block_mask` and never follow the thread; on a chip with no gather hardware this is a real design question.
4. **Dynamic shapes / bucketing — absent, and it is on the critical path.** For a static-shape inference accelerator, *choosing the bucket/pad set* is a cost-model decision with an SLO objective. Add **DietCode** (MLSys'22 — cost model + codegen over shape buckets), Nimble (MLSys'21), Relax/TVM Unity, SoD² (ASPLOS'24), AWS Neuron bucketing, vLLM's shape buckets. LENS's "latency is a step function because of bucketing" is your only hint and you do not follow it.
5. **Distributed / multi-chip — declared out of scope only implicitly.** Either say "excluded, per the 1H-2026 roadmap" in §0, or cover Alpa, GSPMD/Shardy, PartIR, nnScaler, and the comm/compute-overlap compilers (CoCoNet ASPLOS'22, T3 ASPLOS'24, Centauri ASPLOS'24, Flux, TileLink). Right now §4's Cursor/AsyncTP material is orphaned.
6. **Numerics and correctness.** On an fp16-native chip, online softmax rescaling, accumulate width, and FA4's *numerically*-motivated conditional rescaling are compiler concerns. Nothing on accuracy-aware compilation, DL-compiler bug studies, or fuzzing (NNSmith, Tzer). AutoMegaKernel's validator is the lone mention and it is about deadlock, not numerics.
7. **Energy / EDP as an objective.** FFM, Timeloop, ZigZag, Stream, Interstellar all optimize energy or EDP; your review silently converts everything to latency. Spyre is a 75 W part — say why latency-only is the right objective, or add the axis.
8. **Compile time as a first-class objective.** TelaMalloc's headline is compile time, Hidet's and Roller's too, and the roadmap gives a hard "few minutes per model" budget. It is a constraint in Q3's experiment but never a modeled axis.

---

## D. Are the open questions open?

| Q | Verdict |
|---|---|
| **Q1** | **Narrow it.** Ranking-vs-measured on a non-GPU accelerator behind a proprietary compiler exists: TpuGraphs tile collection + Kaufman MLSys'21 (learned, Kendall τ, XLA:TPU), TileLoom's top-2 on Tenstorrent. Surviving novelty: *analytical* model, explicit-NoC dataflow chip, joint tile × work-division space. Also pre-register the repeat/noise protocol — a rank metric computed on singleton measurements is the exact failure mode you already hit. |
| **Q2** | **Genuinely open**, cheapest, best-posed. Keep. |
| **Q3** | **Not open as stated.** "Nobody has published a system that co-optimizes tile shape and scratchpad allocation" is contradicted by Welder (measured, GPU), AKG/PPCG promotion (measured, Ascend), CoSA (MIP), Stream/DeFiNES (simulated). Restate as: joint tile + LX under a **measured** objective on a **per-core-partitioned** scratchpad with work division already fixed — that partition is the real novelty and it is the part you under-sell. |
| **Q4** | **Open.** Strengthen: add Ring/Blockwise attention next to FlatAttention (same "keep K/V on-chip via the fabric" idea), and add the numerics sub-question (does an fp16-accumulate chip even permit the online-softmax recurrence at the seq lengths in the 7 priority models?). |
| **Q5** | **Partially answered elsewhere** — XLA PriorityFusion, Welder, Apollo, AStitch all rank fusion by predicted runtime. Reframe as an Inductor-integration question. Delete the two "determine first" sub-tasks (answered in A4). |
| **Q6** | **Open and cheap.** Add DeFiNES — it already emits capacity-vs-traffic curves for *fused* layer stacks and would give you the fused arm of the experiment nearly for free; Orojenesis alone gives the unfused arm. |

**Missing questions worth adding.** (a) *Bucket and pad selection under a serving SLO* — static shapes are mandatory, vllm-spyre defines the buckets, and no cited system chooses them with a cost model; this is more immediately fundable than Q4. (b) *FP8 block-scale layout × stick alignment* — on the roadmap, zero literature coverage, and a clean layout-algebra question. (c) *Rank quality under a compile-time budget* — every system you cite reports rank quality or compile time, never the Pareto front between them, and the roadmap gives you a hard budget.