# Hot Experts architecture

Inherited foundation: wackMall's resident GPU experts, CPU cold operators and result join.

This fork adds mixed placement, memory reclamation, protected staging/fused submission, asynchronous result uploads and bounded adaptation. [CHANGES](CHANGES.md) records attribution; [BENCHMARKS](BENCHMARKS.md) records results.

## Residency and execution

`--n-cpu-moe N` places expert weights of the first N layers on CPU. With `-ngl all`, the remaining expert layers and eligible non-expert tensors stay GPU-resident. This fork respects that placement instead of forcing all experts to CPU when tiering is enabled.

Large prompt graphs bypass the hot tier and follow static placement. Retaining full GPU expert layers is therefore necessary for the tested prefill profile.

For each eligible host expert tensor, the tier allocates a quantization-preserving GPU hot store, GPU expert-ID lookup table, and CPU cold mask/counters. Gate/up/down weights share a layer's expert ownership. Fully GPU-resident expert tensors do not receive duplicate hot stores. Original CPU weights remain available for cold execution and large prompt graphs; GPU caching alone does not reduce their RAM backing.

S is a per-layer hot-slot budget, not a global LRU capacity and not server parallelism (`-np`). An extra zero-weight sentinel slot maps cold selections to zero in the hot matmul. The CPU mask excludes hot selections, avoiding duplicate contributions. Static seeded stores can have fewer populated slots than requested if their seed is sparse; adaptive stores reserve S usable slots plus the sentinel.

For eligible small graphs:

1. The normal model router selects experts; caching does not alter routing.
2. Resident selections use GPU hot matmuls. Misses use original host weights in CPU cold operators.
3. Eligible MoE blocks use the inherited fused CPU `MOE_COLD` operator; other supported cases use per-matmul cold operators.
4. CPU and GPU contributions are added before the normal routed-weight reduction.

Decode transfers inputs/IDs to CPU and cold results to GPU. Expert weights transfer at initialization and replacement, not on each miss.

Graphs above `LLAMA_EXPERT_TMAX` (default 16 tokens) use the ordinary statically placed path, without tier counting/updates. Small prompt graphs may still use tiered execution: graph size is not generation provenance. The new periodic policy excludes them from learning using explicit attribution. Fused CPU eligibility excludes unsupported bias/scale/activation layouts; the separate hot/cold path or stock path remains available where applicable.

## Selection and replacement

Initial residency comes from a `layer,expert,count` heat CSV, then a compatible score sidecar if no HOT seed is supplied. Without usable heat, adaptation-on initialization starts with low-numbered experts; adaptation-off may leave unseeded layers without stores. A warm seed changes startup behavior and must be accounted for in comparisons.

Two policies remain:

| Policy | Behavior |
| --- | --- |
| Inherited wackMall, `PERIODIC=0` | Decayed counts at eligible graph updates; per-layer replacement with 1.5x score hysteresis and 32-update dwell for occupied slots. Empty slots can fill earlier. |
| This fork, `PERIODIC=1` | Count only attributed one-token generated forwards; apply decay every 32 such forwards, require 64 update opportunities of dwell, and promote at most one expert globally at each boundary. |

Periodic selection chooses the largest eligible score gain, with deterministic layer/expert/slot tie-breaking and finite-score checks. With period 32 and minimum dwell 64, the first initially occupied-slot replacement cannot happen before 2048 attributed forwards. `ADAPT=0` freezes ownership; periodic score collection can still run. Final partial-window counts are drained for persistence without an extra replacement boundary.

`llama_decode_generated()` is a caller-certified API for one text token, sequence 0, in a single-sequence context. The server wires sampled, non-speculative, single-slot generation to it. Ordinary `llama_decode()` is unattributed and does not advance periodic learning. Other clients, including native bench/CLI paths, must not be assumed to exercise bounded adaptation without explicit integration. Speculative and multi-slot adaptation are not validated by the retained profile.

## Pipeline and synchronization

The resident hot/cold split is inherited from wackMall. The following opt-in execution paths were added here for one-token Qwen35MoE graphs with one CUDA backend and synchronous CPU execution:

- Protected input staging (`STAGE_INPUTS=1`) copies layer input/selected IDs into host destinations retained through the join, preventing allocator reuse while cold computation reads them.
- Fused staging (`FUSE_STAGING=1`) submits the eligible GPU prefix and hot suffix with a readiness boundary, making inputs available to CPU cold work while hot work can continue. It reduces submission splitting; it does not imply unrestricted CPU/GPU overlap.
- Guarded cold-result upload (`ASYNC_COLD_UPLOAD=1`) queues protected pinned-host F32 cold outputs on the existing CUDA compute stream before the consuming join. It avoids the synchronous setter's separate completion wait for that eligible copy, not every wait in the graph.

Async upload additionally requires generated-decode attribution, one scheduler copy, two backends, no evaluation callback, successful fused staging and the expected cold-result/join topology. Backend checks cover layout, pinned memory and CUDA device ownership. Other cases retain synchronous fallback. Eligibility currently includes tensor-name/topology checks; this is a focused path, not a generic scheduling interface for arbitrary graphs.

Pending upload ownership survives abort and is drained before scheduler compute reuse, graph allocation, reset or free. Normal scheduler/tier completion waits remain. Replacement synchronizes prior readers, uploads all weight slices, then publishes GPU LUT and CPU mask ownership. No reader may observe a new mapping before its weights are ready.

The `OUTPUTS_BEFORE_SYNC` experiment changes generated-token output-copy ordering relative to the tier barrier. It is off in the measured profile; no dependable throughput gain was established. Host wait/copy/enqueue timers overlap device work and cannot be summed into exclusive inference-time components.

## Memory budget

Automatic S sizing sums one expert slice across eligible host tensors, subtracts a 512 MiB safety allowance from reported free GPU memory, and computes a uniform per-layer budget. Explicit S is clamped to the model's expert count but is not silently reduced to the affordability estimate; allocation failure can fall back to host execution. Neither this estimator nor `-fit off` is a peak-memory watchdog.

Budget includes static model buffers, KV/recurrent state, compute workspaces, hot weights/LUTs, retained staging outputs, CUDA graph/runtime allocations and competing GPU users. Increasing context, cache size or graph retention can change peak usage. Moving more whole layers to CPU frees VRAM but also adds CPU work and changes PP. See [allocation measurements](BENCHMARKS.md#memory-and-correctness).

This fork skips optional Qwen35MoE MTP tensors when MTP loading is disabled and ports compact causal Flash Attention descriptors. Compact mode reuses I64 KV-index information instead of a dense query-by-key mask, only with backend capability and causal/layout checks. Unsupported semantics/layouts retain dense masks. CPU and CUDA consumers were tested; portability is not inferred for every backend.

## Why hit rate is not throughput

Early mixed S18 measured 26.63 TG versus S0's 26.55 despite successful residency. Dependency profiling directed subsequent staging and transfer changes. The longer adaptive replay adds 4.43 percentage points of hot coverage versus its unchanged-seed counterfactual, while paired TG improves only 0.923%. Routing coverage is not a proportional throughput estimate. See [measurements](BENCHMARKS.md#cache-behavior-in-the-longer-replay).

The matched PP1024/TG512 diagnostic reports 17.99% hot coverage over generated-forward selections in eligible CPU-resident layers; the quiet optimized profile measures +14.06% TG versus CPU33/S0. Those selections are not latency-weighted: scheduling, synchronization and data movement determine their critical-path cost. Placement and execution controls also change, so the pair does not isolate any mechanism's contribution. [Diagnostic details](BENCHMARKS.md#same-benchmark-cache-diagnostics-pp1024tg512) distinguish this scope from upstream cache counters.

### Diagnostics and counter scope

`LLAMA_EXPERT_STATS=/path/to/stats.txt` writes exit-time per-layer cold/total selections, repins, periodic decode/boundary counts and coordination/CPU timers (`1` writes to stderr). Hot coverage is `(total - cold) / total` over eligible tiered layers, not all GPU expert work. In periodic mode the cumulative selection counters include attributed generated forwards, not prompt-only observations. `LLAMA_EXPERT_OBSERVE=/path/to/routes.jsonl` records graph provenance and per-layer hot/cold counts plus expert selections; filter `generated_decode=true` for generation coverage. `LLAMA_EXPERT_USAGE=/path/to/heat.csv` exports accumulated per-expert counts for heat/persistence analysis, not a timing report.

Optional RAM-pool `expert_pool` hits refer to the host cold pool, not the GPU hot tier. Startup `seed coverage` is a heat-seed estimate, not live hit rate. Scheduler/expert timing controls inspect staging, uploads and waits, but instrumentation can affect timing and overlapping counters cannot be summed as independent latency. Keep diagnostics in a separate companion and timed runs quiet. [Retained cache measurements](BENCHMARKS.md#cache-behavior-in-the-longer-replay) provide a verified example and its denominator.

The [PP1024/TG512 comparison](BENCHMARKS.md#same-benchmark-cache-diagnostics-pp1024tg512) records upstream and fork counter scopes for the same workload.

## Configuration

Environment controls are read at initialization/graph construction; use a fresh process for controlled changes. CLI spelling is defined in [common/arg.cpp](../common/arg.cpp). These are the fork's key controls, not all llama.cpp options:

| Control | Unset/default behavior | Meaning |
| --- | --- | --- |
| `--n-cpu-moe N` | Explicit placement when supplied | Host expert weights in the first N layers. |
| `--cpu-moe` / `--no-cpu-moe` | Tiering enabled by default | In this fork these enable/disable tiering; unlike newer upstream, `--cpu-moe` alone is not an all-host placement request. Disabling tiering does not undo explicit placement overrides or other fork changes. |
| `LLAMA_EXPERT_S` | Automatic sizing | Uniform hot slots per eligible layer; 0 disables residency. |
| `LLAMA_EXPERT_TMAX` | 16 | Maximum graph token count for the tier path, not the thread count. |
| `LLAMA_EXPERT_HOT` | No CSV | Model-compatible heat seed; overrides sidecar input. |
| `LLAMA_EXPERT_ADAPT` | 1 | Online replacement; 0 freezes ownership. |
| `LLAMA_EXPERT_PERIODIC` | 0 | Opt-in bounded generated-decode policy. |
| `LLAMA_EXPERT_DECAY` | 0.999 | Decay per legacy update or periodic boundary; 1 disables aging. |
| `LLAMA_EXPERT_STAGE_INPUTS` | 0 | Protected one-token input/ID staging. |
| `LLAMA_EXPERT_FUSE_STAGING` | 0 | Eligible fused CUDA prefix/hot submission. |
| `LLAMA_EXPERT_ASYNC_COLD_UPLOAD` | 0 | Guarded async cold-result upload; requires staged/fused eligible path. |
| `LLAMA_EXPERT_OUTPUTS_BEFORE_SYNC` | 0 | Experimental output-copy ordering. |
| `LLAMA_EXPERT_SIDECAR_IN/OUT` | `<model>.tier` | Independent score load/save paths; empty disables that direction. |
| `LLAMA_EXPERT_STATS` | Off | Exit stats: `1` means stderr, otherwise a file path. |
| `LLAMA_EXPERT_USAGE` / `OBSERVE` | Off | Heat CSV / per-layer completed-count JSONL output. |

Sidecars are model-fingerprinted and use physical model-layer indexing, including fully GPU-resident layers. For an isolated cold run, explicitly empty both directions. To resume learning, unset HOT and read a completed prior output with a separate writable output path. A score sidecar is not a KV/prompt cache or a snapshot of exact slot ownership.

Inherited RAM/disk/prefetch controls remain available: `RAMPOOL=0` GiB, `PREAD=1`, `MADVISE=1`, `PREDICT=0`, `PREFETCH_GB=0`, `PREFETCH_MB=64` MiB and `PREFETCH_THREADS=2` (clamped 1..8), all under `LLAMA_EXPERT_`. `PREDICT_LOG` writes prediction and route traces. These mechanisms are not systematically benchmarked in the current profile. Periodic mode requires RAMPOOL/PREAD/MADVISE/PREDICT/PREFETCH_GB off; conflicting settings disable tier initialization with a diagnostic. The reproduction profile also zeros PREFETCH_MB.

`GGML_SCHED_PROFILE`, `GGML_SCHED_PROFILE_DEPS`, `GGML_SCHED_EXPERT_TIMING` and `GGML_SCHED_EXPERT_INPUT_TIMING` enable different diagnostic views. Keep profiling and output logs off for throughput comparisons. Runtime `--cache-ram` is a prompt-cache budget, unrelated to hot-expert slots. The execution switches default off.

## Source map and boundaries

| Area | Source |
| --- | --- |
| Residency, initialization, scoring, persistence | [llama-expert-tier.cpp](../src/llama-expert-tier.cpp) |
| Bounded replacement/provenance helpers | [llama-expert-policy.h](../src/llama-expert-policy.h) |
| Hot/cold graph and staging | [llama-graph.cpp](../src/llama-graph.cpp) |
| Scheduler transfers, fused path, pending ownership | [ggml-backend.cpp](../ggml/src/ggml-backend.cpp) |
| CUDA readiness/upload support | [ggml-cuda.cu](../ggml/src/ggml-cuda/ggml-cuda.cu) |
| CPU cold operators | [ops.cpp](../ggml/src/ggml-cpu/ops.cpp) |
| Attribution integration | [llama-context.cpp](../src/llama-context.cpp), [server-context.cpp](../tools/server/server-context.cpp) |

Rebuild headers, clients and backend libraries together: public model params, generated-decode API, ggml metadata and compact-mask behavior differ from the inherited base. See [validation scope](BENCHMARKS.md#memory-and-correctness) for retained checks and open release gates.
