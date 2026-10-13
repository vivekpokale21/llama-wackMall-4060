# What this fork changes from wackMall

The starting point is miltos22's public wackMall at `39d867bb676a492ed6a667a133d643315ac20474`. Inherited: resident GPU experts with CPU execution for misses. This fork adds mixed placement, memory reclamation and execution changes. [Architecture](HOT_EXPERTS.md) covers operation; [benchmarks](BENCHMARKS.md#from-original-wackmall-to-this-fork) records the starting point and subsequent results.

## Problems addressed on this machine

Original all-CPU placement harmed Tiel Coder PP. Large prompt graphs bypass the hot tier; mixed placement retains full GPU expert layers for those graphs.

The original tier-enabled CLI forced all expert layers to CPU and batch/ubatch 256, overriding the intended CPU35/batch2048 profile. Unused MTP weights and a dense prompt mask consumed memory that could otherwise hold experts. After placement and memory fixes, S18 still measured only 26.63 TG versus 26.55 with S0 on the same early binary. A cache that allocated correctly was not enough: dependency profiling found CPU input handling waiting behind GPU work, so the next changes targeted staging and result transfers. The later fixed S18 three-step result is 31.32 TG.

## Hot Experts and placement

Inherited from wackMall:

- GPU-resident hot stores over host expert tensors, LUT/sentinel remapping, CPU cold masks/operators and CPU/GPU result joins.
- Per-layer decayed scoring and hysteresis/dwell replacement, warm CSV/sidecar seeding, automatic slot budgeting.
- Optional demand RAM pool, mmap page hints, pread staging, router prediction and speculative prefetch. These remain experimental outside the measured hybrid profile.
- CPU fused cold execution and specialized CUDA expert matmul behavior, including hot-store handling.

Added or revised here:

- Preserve explicit `--n-cpu-moe` placement and keep fully GPU-resident expert layers on the normal path. Remove the tier-enable override that forced all expert tensors to CPU and batch/ubatch 256, fit, Flash Attention and KV/parallel settings.
- Honor explicit S within model limits instead of silently reducing it to the affordability estimate. Automatic sizing remains available; allocation failure is still possible.
- Bypass tier counting/updates for large prompt graphs. Add explicit generated-decode provenance so small prompt/speculative/unknown work does not advance the new periodic policy.
- Add opt-in bounded periodic adaptation, global promotion selection and score validation. The inherited policy remains the default.
- Separate sidecar input/output paths, model-layer indexing and controlled score persistence. Empty paths explicitly disable load/save; HOT still takes precedence.

## Pipeline and transfers

- Protected one-token Qwen35MoE input/ID staging, with retained destinations through the cold-result join.
- Opt-in fused CUDA prefix/hot submission with a readiness boundary and fallback when eligibility is not met.
- Opt-in protected pinned-host cold-result upload on the existing CUDA compute stream. Scheduler pending ownership drains before reuse/allocation/reset/free and survives abort recovery.
- A default-off output-copy ordering experiment. No dependable benefit was established, so it is excluded from the measured optimized profile.

Completion, source-lifetime and weight/LUT publication barriers are retained. Cold expert matmuls remain on CPU. Optimized eligibility is narrower than the inherited generic MoE hooks.

## Memory and model loading

- Add `llama_model_params.load_mtp`, default false. Qwen35MoE optional MTP tensors are skipped when unused; common initialization enables loading for draft-MTP. Measured saving: 371.36 MiB for the tested Tiel GGUF, not all models.
- Port compact causal Flash Attention behavior from newer llama.cpp, using commit `5f70b810f2d0e95b75feba1d1be8cd7bf3d0bc31` as a specification rather than blindly cherry-picking. This includes I64 descriptors, KV/graph eligibility and reuse checks, CPU/CUDA consumers, backend support checks and dense fallback.
- Retained graph-output/staging lifetimes can affect workspace allocation. Allocation evidence is profile-specific; generic scratch behavior also exists in upstream and is not assumed removable cache overhead.

## API, diagnostics and tests

- New caller-certified `llama_decode_generated()` and scheduler attribution, integrated into supported single-slot non-speculative server generation.
- Internal backend/tensor metadata for compact masks, protected execution and diagnostics. Rebuild embedders/bindings/backends; ABI compatibility with stock or earlier builds is not promised.
- Opt-in wait/copy/enqueue/dependency/dispatch/hot/cold/join profiling, CPU cold timings, completed-count observations and promotion logs. Timing intervals can overlap; diagnostic totals are not an exclusive latency decomposition.
- Focused policy/provenance, compact-mask, mixed-placement persistence, staging/fusion, output-publication and pending-upload/abort/recovery tests; guarded CUDA MMQ numerical regression.
- Release assertions retained in focused tests, output-environment isolation and CMake/CTest registration. Model-dependent output-sync remains build-only; the prefix-event probe remains exploratory. Focused CUDA runtime checks passed; full automatic CTest and clean-checkout release validation remain open.

## Attribution and maintenance

llama.cpp/ggml provide the runtime and backend foundation; wackMall provides the resident-expert design and inherited mechanisms. This fork's contribution is the bounded integration, execution/lifetime work, memory port, repair and validation above, not every difference between wackMall and upstream. The newer upstream cache comparison is an experiment, not a port of that cache into this tree.

Original [LICENSE](../LICENSE) and [NOTICE](../NOTICE) are unchanged. The historical [wackMall architecture](../ARCHITECTURE.md) is retained.
