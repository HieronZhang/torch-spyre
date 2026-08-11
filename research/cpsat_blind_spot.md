# Where CP-SAT cannot pick the right LX allocation

**Result.** Across five real-model programs and 419 contested tile configurations, CP-SAT's
byte objective is **never strictly worse** than the time optimum. But on two of the programs
it is **degenerate**: several allocations tie exactly on bytes while differing in predicted
transport time by up to **33.8 %**. CP-SAT is free to return any of them. A time-ranked
search is not.

That is a different claim from "CP-SAT is suboptimal", and a more useful one. The byte
objective is not approximately right and occasionally beaten — it is *exactly blind* to a
distinction worth a third of the spill traffic, and no amount of solver effort fixes it,
because the two allocations are equal in the quantity being solved for.

**Status.** Analytic screen, not measured, and not yet compiled. §4 is explicit about what
that does and does not license.

```sh
python3 research/screen_configs.py
```

---

## 1. Why a byte objective can be blind

CP-SAT maximises retained `(read_count + is_intermediate) · size`
(`ilp_solver_ortools.py:208-224`). The cost model divides the same traffic by the rate it
moves at. Dividing every candidate's score by one constant cannot reorder them, so **the two
objectives coincide exactly whenever all contending buffers share a bandwidth.**

They can differ only where rates differ. In the recorded database they never do — of the 98
bundles where LX capacity binds, every single op carries the `default` pattern:

```
access patterns across all binding bundles:   default 265
```

That is the whole reason the earlier corpus found CP-SAT optimal on softmax (36/37) and on
flash attention (3/3). It was not luck. A single-rate bundle makes the byte objective
provably time-optimal.

```sh
python3 research/find_cpsat_gaps.py --records <db.json>
```

## 2. Flash attention has the spread and still cannot use it

Flash is the one recorded workload that mixes transports — three of its movable buffers are
`cat` outputs tagged `stick_scatter`:

| config | slow buffers | their rate | default |
|---|---|---:|---:|
| `Lq=Lk=4096 h8 q4` | `b1`, `b3`, `b19` | 105.6–115.2 GB/s | 150 |
| `Lq=Lk=2048 h8 q1` | `b1`, `b3`, `b19` | 108.0–117.6 GB/s | 150 |

So why is CP-SAT still exactly optimal there? Because those slow buffers are also its
**largest** — 1024 KB and 512 KB per core. Both objectives want to keep them, and agree.

Writing the condition down explains it. With `A` slow and `B` fast, a time-ranked search
beats a byte-ranked one only when

```
bw_A / bw_B  <  weight_A / weight_B  <  1
```

— the slow buffer must be the **cheaper** one by the byte objective. Flash sits above the
window, not inside it. That is a structural fact about the workload, not a limit of the
search.

## 3. A program that lands inside the window

`prefix_block` (`research/workloads.py`) is a chunked-prefill step: concatenate the cached
prefix with new hidden states along the sequence dimension, RMS-normalise, then the SwiGLU
up-projection. The concat is on a partition dimension, so the extractor tags it
`stick_scatter` (`dump_cost_model.py:442`), and its stick dimension is `d_model` = 4096 —
which is what drives the rate down:

| buffer | per-core | consumers | rate | byte weight | time weight |
|---|---:|---:|---:|---:|---:|
| `gate` | 1600 KB | 1 | 150.0 | 3277 K | 21.8 |
| `up` | 1600 KB | 1 | 150.0 | 3277 K | 21.8 |
| **`h`** (the concat) | 1024 KB | 2 | **62.4** | **3146 K** | **50.4** |
| **`xn`** (the rescale) | 1024 KB | 2 | 118.0 | **3146 K** | 26.7 |

`h` and `xn` carry the same dimensions and the same number of consumers, so **CP-SAT scores
them identically — 3146 K each.** Its optimum is a tie. But `h` is a concat at 62.4 GB/s and
`xn` a broadcast at 118, so spilling `h` costs nearly twice what spilling `xn` costs.

At `bt=1 st=1 ft=2` on 32 cores, the two allocations achieving CP-SAT's optimum are:

```
keeps ['h', 'inv', 'ms']    spill time 4,138,380      <- also the time optimum
keeps ['inv', 'ms', 'xn']   spill time 6,022,590      +33.8%
```

Eight configurations behave this way, spanning +17.7 % to +33.8 %:

| knobs | cores | spread |
|---|---:|---:|
| `bt=1 st=1 ft=2` | 32 | +33.8 % |
| `bt=4 st=1 ft=2` | 8 | +33.8 % |
| `bt=2 st=2 ft=2` | 8 | +31.1 % |
| `bt=1 st=4 ft=2` | 8 | +28.6 % |
| `bt=1 st=1 ft=1` | 32 | +20.8 % |
| `bt=4 st=1 ft=1` | 8 | +20.8 % |
| `bt=2 st=2 ft=1` | 8 | +19.2 % |
| `bt=1 st=4 ft=1` | 8 | +17.7 % |

`decode_block` — the same idea with the concat on the KV cache — shows the effect at 1.1–2.8 %
only. Its concat's stick dimension is `head_dim` = 128 rather than `d_model` = 4096, giving
~110 GB/s instead of ~62. **The width of the concatenated tensor is what decides whether this
matters**, which is not obvious until the rate formula is written out.

## 4. What this does not establish

- **Nothing is measured.** No device has run any of these programs.
- **Nothing is compiled.** The screen models spilled traffic analytically from tensor shapes
  and the model's own rate formulas. It does not run the extractor, so `h` being tagged
  `stick_scatter` is a prediction from `dump_cost_model.py:442`, not an observation.
- **The percentages are spill traffic, not runtime.** A 33.8 % spread in transport time is a
  smaller share of total runtime, and how much smaller depends on the compute the program
  also does. Only the real cost model on real features gives that.
- **CP-SAT's tie-break is unknown.** OR-Tools may return either tied allocation and nothing
  here says which. It may already pick well by accident. That is exactly the point: it is
  unspecified, so it cannot be relied on.
- **`bw_peak_gbps` = 150 and the cat0 surface are fitted constants.** The 1.9× ratio between
  `h` and `xn` inherits their fit error.

## 5. The experiment that would settle it

Compile `prefix_block` at the eight configurations above and confirm three things in order.
Each can falsify the claim on its own:

1. **Does `h` get tagged `stick_scatter`?** If the extractor does not tag the concat, the
   rate difference never appears and the case evaporates.
2. **Does CP-SAT's optimum really tie?** Read the retained set out of `SPYRE_DUMP_COST=1` and
   check whether `h` or `xn` was kept.
3. **How much runtime is it worth?** Run both allocations and measure.

```sh
IS_INDUCTOR_SPAWNED_SUBPROCESS=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 SPYRE_DUMP_COST=1 \
  python3 research/compile_only.py --workload prefix_block --bt 1 --st 1 --ft 2 --cores 32
```

If step 1 fails, the fix is not to abandon the direction but to find the concat shape the
extractor does tag — the tagging rule requires a device dimension below 64 just inside the
stick, and `prefix_block`'s exact device layout has not been checked against it.

## 6. What follows

The actionable form of this is narrower than a cost-model allocator and much cheaper: **give
CP-SAT a tie-break on transport rate.** `spill_cost()` (`ilp_solver_ortools.py:208`) already
returns the per-buffer objective; dividing it by the op's effective bandwidth turns the
degenerate tie into a strict preference, without changing the solver, the capacity model, or
anything else about the pass. That is a one-function change whose worst case is the behaviour
CP-SAT has today.

It should not be made on the strength of this document alone. §5 comes first.
