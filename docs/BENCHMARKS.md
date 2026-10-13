# Benchmarks and experiments

Laptop-specific measurements. Deltas use each table's stated baseline; percentages use displayed rates unless noted. [Architecture](HOT_EXPERTS.md) | [Changes](CHANGES.md).

## Results overview

- [Same-binary PP1024/TG512](#fresh-same-binary-pp1024tg512): optimized CPU35/S18 versus CPU33/S0 gives +14.06% warm TG, -4.41% warm PP, -10.68% warm prompt/decode time and +290 MiB peak VRAM. Placement, residency and execution settings change together; this is not an isolated cache-benefit estimate.
- [Internal three-step agentic benchmark](#internal-three-step-benchmark-results): fixed and adaptive S18 measure +11.56% and +11.99% TG versus the same-fork CPU33/S0 control. Agent trajectories differ.
- [Matched cache diagnostics](#same-benchmark-cache-diagnostics-pp1024tg512) report 17.99% fork hot coverage; upstream counters have different scopes. [Earlier cache experiments](#earlier-expert-cache-experiments) did not consistently reproduce advertised gains on this hardware and workload.

## Hardware and model scope

- ASUS ROG Zephyrus G14 (2023), AMD Ryzen 9 7940HS (8 cores / 16 threads).
- NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM; AMD Radeon 780M handles display output.
- 32 GB DDR5, Windows 11 + WSL2. The current measured hybrid profile uses CUDA 12.4.131, GNU 13.3.0, Release, SM89 and CUDA graphs.
- Daily-use targets: KAT-Coder-V2.5-Dev, Tiel Coder and Qwen 3.6 35B-A3B-class coding models, primarily Q4/Q5 GGUFs.
- The numerical comparisons below use `Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`, identified by the loader as `qwen35moe`. The GGUF includes Q5_K/Q6_K expert tensors.

Development and benchmarking primarily target this configuration. KAT and other Q4/Q5 variants do not have a comparable fork benchmark series documented here. Other hardware and architectures are not systematically tested, rather than declared unsupported. Laptop power/thermal settings, driver version and full GGUF checksum should be captured with future published runs; they are not uniformly recorded in the summarized evidence.

Tiel Coder is the daily-use agentic coding model behind this benchmark series.

## PP1024/TG512 benchmarks

Primary comparison: fixed work, with first-request and warm rates separated. The internal three-step benchmark is secondary.

### Fresh same-binary PP1024/TG512

Both profiles use the same repaired CUDA 12.4/SM89 server and libraries. Shared launch parameters:

```bash
unset CUDA_LAUNCH_BLOCKING
# MODEL: Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
./build-cuda124-sm89/bin/llama-server \
  -m "$MODEL" --host 127.0.0.1 --port 8080 --jinja --no-mmproj \
  -c 110000 -ngl all --n-cpu-moe "$CPU_MOE" -b 2048 -ub 2048 -np 1 \
  -ctk q8_0 -ctv q8_0 --flash-attn on --no-kv-unified --load-mode none -fit off \
  -t 8 -tb 8 -td 8 -tbd 8 --cache-ram 0 --no-warmup --spec-type none
```

Profile-specific settings (export the `LLAMA_EXPERT_*` values before launch):

| Parameter | Static control | Optimized hybrid |
| --- | --- | --- |
| `CPU_MOE` | `33` | `35` |
| `LLAMA_EXPERT_S` | `0` | `18` |
| `LLAMA_EXPERT_ADAPT` | `0` | `1` |
| `LLAMA_EXPERT_STAGE_INPUTS` | `0` | `1` |
| `LLAMA_EXPERT_FUSE_STAGING` | `0` | `1` |
| `LLAMA_EXPERT_ASYNC_COLD_UPLOAD` | `0` | `1` |

The hybrid uses the periodic policy and retained warm seed; remaining environment settings and path placeholders are in [Reproducing the tested profile](#reproducing-the-tested-profile). Request sampling is specified below, not inherited from the agentic benchmark's server defaults.

Each timed server receives the same frozen 1024 prompt token IDs five times, with 512 generated tokens per request, greedy temperature 0, seed 1, ignore EOS and no prefix reuse. Separate preflight servers pass arithmetic/code checks before either timed arm starts. Every timed response reports 1024 prompt / 512 generated tokens, cache_n=0 and no truncation.

Baseline: CPU33/S0 in this fresh pair; percentage rate deltas use unrounded pooled timings.

| Configuration | First PP / TG | Warm PP | PP delta | Warm TG | TG delta | Warm prompt / generated tokens | Warm prompt + decode seconds | Peak MiB | VRAM delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU33/S0 static | 61.50 / 27.01 | 551.19 | 0.00% | 29.55 | 0.00% | 4,096 / 2,048 | 76.7335 | 7,167 | 0 |
| CPU35/S18 adaptive optimized | 158.02 / 32.31 | 526.85 | -4.41% | 33.71 | +14.06% | 4,096 / 2,048 | 68.5354 | 7,457 | +290 |

Warm prompt/decode time falls 10.68%. Each arm processes 5120 prompt / 2560 generated tokens overall. Warm TG ranges: S0 29.42-29.88; S18 33.21-33.93. PP falls 4.41% in this fixture.

Order: static then hybrid, without a reverse repeat or fresh upstream arm. Placement and execution controls change together. Prompt/decode seconds exclude startup and client overhead; the first-request difference was not isolated.

Both profiles have CUDA compute 423.40 MiB and host compute 32.34 MiB; CUDA model buffers are 5388.80 MiB (CPU33) and 4426.80 MiB (CPU35). No dense 430 MiB prompt mask, allocation fallback, CUDA error or swap-counter delta was observed. Source/status and binary/input hash checks passed; swap was restored after the run.

Driver: NVIDIA-SMI 615.71.08 / KMD 616.92. Live mappings confirmed CUDA 12.4 libraries. Windows power mode/fan profile was not recorded.

Immutable input SHA256 values:

- GGUF: `54f46c4ce544c225122b0f066c2336f10404be7bc53b0cc94d1dbbc5e826bdc1`.
- Heat seed: `003311e6415b71d3b6c173059885910bb69cecd70f2f31e7b6948d66b02a2096`.
- Frozen prompt JSON: `45c0f29beb4ae924d2112aef0e28a9f054d3e6bf1c40677f2c3f8edd4a29c2e4`.


The separate matching diagnostic below found no replacements during these five requests; adaptation was enabled, but the short-run uplift is not evidence of adaptive replacement improving TG.

### Server-driven PP1024/TG512

The later comparator is pinned llama.cpp merge `d6cf9acb25e6c657d6c6a4645ab7b69fce119995` (retained PR29887 comparison). It has shared LRU banks and uploads eligible misses for GPU execution. Its `--moe-cache-mib` budget and shared slots are not equivalent to this fork's per-layer S. All-CPU placement describes host weight residency, not CPU-only expert execution.

Five requests per fresh server, identical frozen 1024 prompt token IDs, exactly 512 generated tokens per request, no prefix reuse. Greedy temperature 0, seed 1, ignore EOS; no preceding completion or tokenizer request. First request is separate; warm rows pool four requests (4096 prompt / 2048 generated tokens).

Baseline: pinned upstream CPU35/cache off. Warm rates exclude the first request.

| Configuration | First PP / TG | Warm PP | PP delta | Warm TG | TG delta | Warm prompt / generated tokens | Peak MiB | VRAM delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream CPU35, cache off | 127.08 / 23.11 | 493.52 | 0.00% | 27.79 | 0.00% | 4,096 / 2,048 | 6,825 | 0 |
| Upstream CPU35, cache 640 MiB | 184.69 / 17.29 | 497.65 | +0.84% | 18.77 | -32.46% | 4,096 / 2,048 | 7,467 | +642 |
| Upstream all CPU MoE, cache 2944 MiB | 22.29 / 13.06 | 432.66 | -12.33% | 21.89 | -21.23% | 4,096 / 2,048 | 7,383 | +558 |
| Repaired fork CPU35/S18, adaptation on | 26.97 / 27.57 | 502.19 | +1.76% | 33.27 | +19.72% | 4,096 / 2,048 | 7,457 | +632 |

The rows are retained separate runs, not a fresh interleaved/reversed-order four-way series. The fork row uses the isolated validated MMQ scratch-padding repair, not the older default build. That default build failed before completing the first prefill under both S0 and S18; failed timings are excluded. The repair passed all five requests, arithmetic/code checks, nine guarded numerical cases and 37 CUDA/CPU comparisons. The fresh CPU33/S0 versus S18 pair above subsequently checked the same repaired server; a full-task rerun of that repaired binary is not recorded here.

Small-batch upstream cache counters across five requests: 38.23% hits at CPU35/cache640 and 62.46% at all-CPU/cache2944, with approximately 795.77 GiB and 556.05 GiB uploaded respectively. These are micro-run counters, not live-task hit rates or exclusive PCIe transfer time. A larger cache helped relative to the smaller configuration, but static placement changed too. The retained fork warm TG is 52.00% above the all-CPU upstream profile; this is configuration-specific, not a universal cache speedup.

### Same-benchmark cache diagnostics: PP1024/TG512

A fresh fork diagnostic repeats exactly the five-request PP1024/TG512 sequence used by the retained upstream runs (640 MiB and 2944 MiB). All three use the identical frozen prompt SHA256 recorded above, 5120 processed prompt / 2560 returned generated tokens overall, temperature 0, seed 1, ignore EOS and no prefix reuse. Model, requested context, Q8 K/V, Flash Attention, batch/ubatch and threads match. Generated token IDs are not forced identical across forks. The upstream rows are older runs of the same benchmark, not a new paired run.

Performance baseline: upstream CPU35/cache640, shown first. Rates are from separate quiet runs of the same benchmark; counter definitions still differ.

| Configuration | Cache-counter scope | Hits / misses | Reported or derived hit rate | Quiet warm PP | PP delta | Quiet warm TG | TG delta | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream CPU35, cache 640 MiB | Small-batch lookups, 35 layers | 275,378 / 445,004 | 38.23% | 497.65 | 0.00% | 18.77 | 0.00% | 7,467 |
| Upstream all CPU MoE, cache 2944 MiB | Small-batch lookups, 40 layers | 514,422 / 309,141 | 62.46% | 432.66 | -13.06% | 21.89 | +16.62% | 7,383 |
| Fork CPU35/S18, ADAPT=1 optimized | Generated-forward selections, 35 layers | 128,670 / 586,730 | 17.99% | 526.85 | +5.87% | 33.71 | +79.60% | 7,457 |

Hit/miss counters cover the whole five-request sequence, including its first request; quiet warm rates pool the remaining four requests (4096 prompt / 2048 generated tokens). Fork rates are from the uninstrumented same-binary pair, not the diagnostic's timings. The diagnostic's five outputs and final score sidecar match that quiet S18 run exactly; peak is 7457 MiB, allocation/correctness and no-swap/error/fallback gates pass. Source/status and binary/input checks pass before/after; swap is restored and GPU usage returns to zero.

The rates still have different definitions. Upstream deduplicates expert IDs within each layer's cache lookup and logs hits/misses for batches of up to eight tokens; those counters are not explicitly generated-forward-only. The fork counts every selected expert in attributed generated forwards: 2555 forwards x 35 eligible layers x 8 selections = 715,400. Returning 512 tokens takes 511 such forwards per request; the first output comes from prefill. Six non-generated graphs are excluded from its hit rate. Neither row includes statically GPU-resident layers as cache hits. Placement, capacity and initialization also differ: the fork starts with its retained heat seed, while upstream starts with an empty shared LRU.

The fork completes 79 policy boundaries but **zero repins**, hence zero replacement-weight bytes. Its dwell clock reached the first possible replacement boundary, but a boundary does not guarantee an eligible promotion. This is a seeded-residency/execution measurement, not short-run evidence of adaptation benefit. Zero replacement bytes does not mean zero H2D traffic: initial hot-store loading and cold-result uploads are outside that counter. Upstream records 795.77 GiB and 556.05 GiB of small-batch weight uploads respectively; its misses upload weights for GPU execution, whereas fork misses compute on CPU. The higher upstream hit rates therefore do not establish lower latency, and these byte counters do not measure PCIe saturation or exclusive transfer time.

Do not substitute the longer replay's 20.93% hot coverage into this comparison. Likewise, no matching live three-step cache-hit comparison is available: upstream task logs did not retain exit counters. The tables above separate benchmark workloads rather than borrowing counters across them.


Profiling the upstream all-CPU path found substantial H2D traffic and reported a Gen4 x8 link, not x16. A later 21.53-second trace recorded 8.28 seconds external H2D and 6.65 seconds graph execution; graph duration includes internal operations/transfers/gaps, not pure arithmetic. Collection/version warnings qualify those traces. Synchronization overlaps device execution, so these totals neither prove continuous PCIe saturation nor identify independently removable wait time.

## Internal three-step benchmark results

Internal three-step agentic coding benchmark on my custom Pi fork. All rows pass all three steps and both validations. Rates pool raw server timings; agent trajectories differ. Inputs and harness are private.

### Same-fork static and hybrid controls

Same fork binary, Tiel Q4_K_XL, requested context 110000, Q8 K/V, Flash Attention, batch/ubatch 2048, one server slot, eight threads, MTP off, fit off. S18 uses the retained warm heat seed and staging/fusion/async-upload profile.

Baseline: same-fork CPU33/S0. TG deltas use unrounded retained rates; live-trajectory differences still apply.

| Configuration | PP tok/s | PP delta | TG tok/s | TG delta | Prompt / generated tokens | Calls | Task seconds | Peak MiB | VRAM delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU-MoE 33, S=0 | 441.83 | 0.00% | 28.07 | 0.00% | 33,389 / 3,609 | 19 | 309.231 | 7,219 | 0 |
| CPU-MoE 35, S18, fixed ownership | 397.87 | -9.95% | 31.32 | +11.56% | 38,607 / 4,046 | 19 | 306.694 | 7,493 | +274 |
| CPU-MoE 35, S18, periodic adaptation | 474.11 | +7.31% | 31.44 | +11.99% | 35,682 / 3,465 | 17 | 262.977 | 7,477 | +258 |

Adaptation adds 0.389% TG over fixed S18 in this live pair. Different prompts, outputs and agent paths prevent fixed-work attribution of PP or task duration.

### Newer upstream weight-upload cache

The later comparator is pinned llama.cpp merge `d6cf9acb25e6c657d6c6a4645ab7b69fce119995` (retained PR29887 comparison). It has shared LRU banks and uploads eligible misses for GPU execution. Its `--moe-cache-mib` budget and shared slots are not equivalent to this fork's per-layer S. All-CPU placement describes host weight residency, not CPU-only expert execution.

Live three-step observations, same Tiel/110000/Q8/FA/batch/thread profile:

Baseline: pinned upstream CPU35/cache off. Placement, budget and live trajectories differ.

| Configuration | PP tok/s | PP delta | TG tok/s | TG delta | Prompt / generated tokens | Peak MiB | VRAM delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream CPU35, cache off | 460.23 | 0.00% | 26.60 | 0.00% | 37,548 / 3,428 | 6,851 | 0 |
| Upstream CPU35, cache 640 MiB | 384.17 | -16.53% | 16.65 | -37.41% | 27,510 / 4,295 | 7,505 | +654 |
| Upstream all CPU MoE, cache 2944 MiB | 338.79 | -26.39% | 20.05 | -24.62% | 37,872 / 3,863 | 7,423 | +572 |
| Fork CPU35/S18, adaptation on | 474.11 | +3.02% | 31.4385 | +18.19% | 35,682 / 3,465 | 7,477 | +626 |

All pass the three steps and both validations. The 640 MiB live arm exceeds the preferred 7500 MiB limit by 5 MiB, but not the hard 7680 MiB guard. The all-CPU arm is a separate later run; no all-CPU/cache-off live control was measured. Placement, budget, seed initialization, base revision and sampled trajectories differ. These results are not an equal-cache-budget comparison.

## Fixed-work and longer validation

Two ordered short async-upload off/on pairs used the same fork binary, S18 fixed ownership, frozen 7500-token empty-prefix and 4738-new-token reused-suffix prompts, then 128 generated tokens per cycle. Outputs matched exactly. Pooled six decode cycles per mode: TG 31.62 -> 32.68 (+3.35%); warm empty PP 737.57 -> 725.82 (-1.59%), reused PP 646.28 -> 642.02 (-0.66%). Peak remained 7495 MiB. This isolates the guarded upload switch within that fixture, not the entire fork.

A longer fixed/adaptive replay used 64 requests and 8192 generated tokens per run, both process orders, and 267510 main prompt tokens per mode across the pooled pair:

Baseline: fixed S18. Each mode pools two 64-request runs; VRAM is the maximum peak across its two runs.

| Mode | Pooled TG | TG delta | Warm empty PP | Empty PP delta | Warm reused PP | Reused PP delta | Main prompt / generated tokens | Main request seconds | Peak MiB (max) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed S18 | 32.3476 | 0.00% | 725.9923 | 0.00% | 623.0444 | 0.00% | 267,510 / 16,384 | 879.3111 | 7,497 |
| Adaptive S18 | 32.6461 | +0.92% | 712.2642 | -1.89% | 631.2158 | +1.31% | 267,510 / 16,384 | 875.2469 | 7,495 |

Adaptation reduces pooled request time by 0.462%.

### Cache behavior in the longer replay

The longer replay above has a separate instrumented adaptive companion with `LLAMA_EXPERT_STATS` and `LLAMA_EXPERT_OBSERVE` enabled. It uses the same Tiel model, CPU35/S18, 110000 context, Q8 K/V, Flash Attention, batch/ubatch 2048, eight threads and optimized staging/fusion/async-upload settings. It predates the later MMQ scratch-padding repair and is not the binary or workload of the fresh PP1024/TG512 pair above. It is relevant evidence for bounded adaptation and residency behavior, not a fresh hit-rate measurement.

Re-counting its raw routing observations reproduces 8130 attributed generated forwards, 73 excluded non-generated graphs and all 35 per-layer cold/total counters. Across eligible CPU-resident layers there are 2,276,400 expert selections (8130 forwards x 35 layers x 8 selections), of which 476,359 are hot: **20.93% runtime hot coverage**. Fully GPU-resident layers are excluded, so this is not the fraction of all model expert work executed on GPU. These are selection counts, not expert-byte or compute-time savings; prompt work and startup seed coverage are excluded.

| Generated-forward window | Actual hot selections / total | Actual hot rate | Unchanged seed rate on the same routing |
| --- | ---: | ---: | ---: |
| Initial, through forward 2048 | 108,041 / 573,440 | 18.84% | 18.84% |
| After first promotion | 368,318 / 1,702,960 | 21.63% | 17.20% |

The initial versus post rate change is not isolated adaptation benefit because routing can change over time. The same-routing seed counterfactual is more specific: after the first promotion, adaptation adds 75,404 hot selections (+4.43 percentage points) and reduces cold selections by 5.35% versus keeping that seed layout.

The companion records 254 policy boundaries and 96 repins, with the first at forward 2048. Promotion transfers total 175.28125 MiB; summed reported promotion-copy time is 45.305 ms. Ownership, eligibility/global ranking, quantized copy bytes and final scores passed the retained independent verifier. The companion's sampled outputs/final scores match both quiet adaptive runs; the quiet paired TG gain remains the separately measured 0.923% above. Diagnostic timing includes instrumentation and overlapping work, so coordination/CPU timers are not an exclusive latency partition.

## From original wackMall to this fork

Development stages from the original GPU-hot / CPU-cold implementation to the optimized mixed-placement profile.

All rows below use the same internal three-step agentic coding benchmark on my custom Pi fork and pass all three steps. The benchmark and its inputs are private and are not included here. These are development stages, not identical agent trajectories or one controlled before/after series.

Baseline: historical CPU33 static reference, shown first. These cross-stage deltas are descriptive, not isolated effects of our changes.

| Stage | PP tok/s | PP delta | TG tok/s | TG delta | Prompt / generated tokens | Peak MiB | VRAM delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical CPU33 static reference | 464.69 | 0.00% | 27.33 | 0.00% | 31,351 / 3,836 | 7,539 | 0 |
| Public wackMall, warm S8 | 116.60 | -74.91% | 23.32 | -14.67% | 39,020 / 3,717 | 4,845 | -2,694 |
| Public wackMall, warm effective S35 | 106.25 | -77.14% | 24.79 | -9.29% | 42,267 / 5,376 | 6,825 | -714 |
| Early fork, placement/memory fixes, CPU35/S0 | 361.08 | -22.30% | 26.55 | -2.85% | 42,198 / 4,855 | 6,257 | -1,282 |
| Early fork, placement/memory fixes, CPU35/S18 fixed | 305.18 | -34.33% | 26.63 | -2.56% | 28,732 / 3,552 | 7,445 | -94 |
| Fork with staging/fusion/async upload, CPU35/S18 fixed | 397.87 | -14.38% | 31.32 | +14.60% | 38,607 / 4,046 | 7,493 | -46 |
| Fork with staging/fusion/async upload, CPU35/S18 adaptive | 474.11 | +2.03% | 31.4385 | +15.03% | 35,682 / 3,465 | 7,477 | -62 |

Public wackMall forced batch/ubatch 256, all 40 expert layers to CPU. At S8 it used a 0.64 GiB hot store and left roughly 3 GiB VRAM free; that low allocation was not an efficiency win. The retained static CPU33 reference reached 27.33 TG / 464.69 PP. Increasing the public cache to effective S35 still did not beat that reference. The high replacement counts (601 at S8, 3277 at S35) and prompt-contaminated routing counters are not evidence of a particular isolated cause.

Poor PP motivated restoring mixed CPU-MoE placement. The batch override and other configuration differences prevent isolating placement's contribution.

The early fork rows use CUDA 12.4, context 110000, batch/ubatch 2048 and CPU35 mixed placement, with the memory ports applied. S0/S18 use the same binary. S18 now allocates successfully, but its TG is only about 0.3% above S0 while consuming 1188 MiB more peak VRAM. Fixing placement and making room for the cache did not by themselves establish a useful cache speedup.

The final adaptive S18 profile includes protected staging, fused submission and asynchronous cold-result uploads. The [fixed-versus-adaptive comparison](#same-fork-static-and-hybrid-controls) is recorded separately above. The entire gain cannot be attributed to scheduling: configurations and agent trajectories differ.

### Project-start references

The retained original reference is cache-disabled `llama.cpp-moe-cache` build 11587, commit `02995f08620600700eef5e743b309c5b4e851c64`, not a pristine upstream checkout. Same model/quant, CUDA 12.4, Q8 K/V, batch/ubatch 2048, eight threads, no MTP; context differs as shown.

Baseline: original CPU33 at context 131072. Context/revision differences still apply.

| Reference | Context | PP tok/s | PP delta | TG tok/s | TG delta | Prompt / generated tokens | Peak MiB | VRAM delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original CPU33 | 131072 | 464.69 | 0.00% | 27.3302 | 0.00% | 31,351 / 3,836 | 7,539 | 0 |
| Original CPU35 | 110000 | 244.27 | -47.43% | 26.2340 | -4.01% | 34,516 / 4,326 | 6,327 | -1,212 |
| Fork adaptive CPU35/S18 | 110000 | 474.11 | +2.03% | 31.4385 | +15.03% | 35,682 / 3,465 | 7,477 | -62 |

Observed total TG gains are 15.03% versus original CPU33 and 19.84% versus CPU35. These are cross-revision/cross-trajectory observations; CPU33 also has a larger context. Slow first requests dominate original CPU35 aggregate PP: excluding the first request gives 435.25 PP versus current 486.74, still not a matched prompt comparison. Do not claim a general PP doubling or attribute total project progress solely to cache replacement.

## Earlier expert-cache experiments

Several existing approaches were evaluated by varying cache size, slot count, CPU-MoE placement and residency. Their advertised gains were not reproduced consistently on this hardware/workload. That is a local observation, not proof that the implementations are flawed.

- Early inherited wackMall TG512 tests on Tiel Q4_K_XL forced all-host expert placement/batch 256. Warm S8 achieved 23.44 TG; effective S35 achieved 19.72 and 19.58 TG despite higher residency/seed coverage. The retained CPU-MoE36 reference was 24.61 TG.
- Post-memory-port fixed-cache live trials: S12 measured 420.29 PP / 26.10 TG / 7093 MiB; S18 measured 305.18 PP / 26.63 TG / 7445 MiB. Different token counts and no matched same-binary S0 in that pair prevent a direct PP conclusion. The initial cache design did not establish the later optimized gains.

These results motivated mixed placement and dependency/copy profiling rather than further cache-capacity tuning.

## Headline summary and attribution

The maintainer supplied this rounded summary. It is kept separate because the inspected artifacts do not identify one exact matched three-arm run with these values:

Baseline: reported upstream CPU33 (~27.0 TG). PP and VRAM are missing for this specific summary, not borrowed from another series.

| Configuration | Reported PP | PP delta | Reported TG | TG delta | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upstream llama.cpp, CPU-MoE 33 | Not recorded | Not recorded | ~27.0 tok/s | 0.00% | Not recorded |
| This fork, CPU-MoE 33, S=0 | Not recorded | Not recorded | ~28.5 tok/s | +5.56% | Not recorded |
| This fork, CPU-MoE 35, S=18 | Not recorded | Not recorded | ~31.5 tok/s | +16.67% | Not recorded |

From those rounded values:

- +5.6% TG: non-cache fork changes versus the reported upstream baseline.
- +10.5% TG: Hot Experts configuration versus this fork's S=0 control.
- +16.7% TG: total reported gain versus upstream, not the isolated effect of caching.

The second comparison also changes static CPU-MoE placement and enables the optimized execution profile; it does not isolate residency, adaptation or one pipeline change. PP is reported as effectively tied between the final configurations. Exact PP values, workload, upstream revision, context, token counts, run dates and peak VRAM for this particular summary still need identification. Do not substitute figures from another run to fill those gaps.

## Reproducing the tested profile

Build instructions are in [README](../README.md#build-and-run). Run from the repository root, with a consistent freshly built library set and a clean environment. Replace the three file paths below. The warm seed is model-specific and is not bundled; an arbitrary seed does not reproduce these rates.

```bash
export MODEL=/path/to/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
export HOT_SEED=/path/to/model-compatible-heat.csv
export TIER_OUT=/path/to/writable/session.tier
unset LLAMA_EXPERT_STATS LLAMA_EXPERT_USAGE LLAMA_EXPERT_OBSERVE LLAMA_EXPERT_PREDICT_LOG
unset GGML_SCHED_EXPERT_TIMING GGML_SCHED_EXPERT_INPUT_TIMING
unset GGML_CUDA_DISABLE_GRAPHS GGML_CUDA_GRAPH_OPT GGML_CUDA_NO_PINNED CUDA_LAUNCH_BLOCKING
export GGML_SCHED_PROFILE=0 GGML_SCHED_PROFILE_DEPS=0
export LLAMA_EXPERT_S=18 LLAMA_EXPERT_TMAX=16
export LLAMA_EXPERT_STAGE_INPUTS=1 LLAMA_EXPERT_FUSE_STAGING=1 LLAMA_EXPERT_ASYNC_COLD_UPLOAD=1
export LLAMA_EXPERT_PERIODIC=1 LLAMA_EXPERT_ADAPT=1 LLAMA_EXPERT_DECAY=0.999
export LLAMA_EXPERT_OUTPUTS_BEFORE_SYNC=0
export LLAMA_EXPERT_RAMPOOL=0 LLAMA_EXPERT_PREAD=0 LLAMA_EXPERT_MADVISE=0
export LLAMA_EXPERT_PREDICT=0 LLAMA_EXPERT_PREFETCH_GB=0 LLAMA_EXPERT_PREFETCH_MB=0
export LLAMA_EXPERT_HOT="$HOT_SEED"
export LLAMA_EXPERT_SIDECAR_IN= LLAMA_EXPERT_SIDECAR_OUT="$TIER_OUT"

./build-cuda124-sm89/bin/llama-server \
  --host 127.0.0.1 --port 8080 -m "$MODEL" --jinja --no-mmproj \
  -c 110000 -ngl all --n-cpu-moe 35 -b 2048 -ub 2048 -np 1 \
  -ctk q8_0 -ctv q8_0 --flash-attn on --no-kv-unified --load-mode none -fit off \
  -t 8 -tb 8 -td 8 -tbd 8 --cache-ram 0 --no-warmup --spec-type none \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --repeat-penalty 1 --presence-penalty 0
```

Verify dynamic libraries resolve to this build and CUDA 12.4, not another toolkit or fork. If setting `GGML_BACKEND_PATH` for an explicit backend override in this base, use the library file (`.../libggml-cuda.so`), not a directory. Set `LD_LIBRARY_PATH` to your build/toolkit library directories if your installation requires it; verify live library mappings rather than relying on the variable alone.

Controls:

- Fixed optimized S18: change only `LLAMA_EXPERT_ADAPT=0`; keep periodic mode, seed and pipeline switches.
- Same-fork static S0: set `LLAMA_EXPERT_S=0`, `LLAMA_EXPERT_ADAPT=0` and staging/fusion/async-upload to 0. Use CPU33 for the best-static placement control or CPU35 for same-placement isolation; label them separately.
- Upstream controls: use the pinned comparator, clear this fork's expert/scheduler/test overrides, and use its cache flag. Upstream all-host placement uses `--cpu-moe`, which has different semantics in this fork.

For a fixed-token microbenchmark, POST to `/completion` with this body, using the retained prompt's 1024 token IDs as `prompt`:

```json
{
  "prompt": ["REPLACE WITH 1024 INTEGER TOKEN IDS"],
  "n_predict": 512,
  "temperature": 0,
  "seed": 1,
  "ignore_eos": true,
  "cache_prompt": false,
  "stream": false
}
```

Replace the placeholder with one frozen integer-token array. Send five sequential requests without warmup/preflight completions to the timed process. Check response `timings.prompt_n=1024`, `predicted_n=512`, `cache_n=0`, no truncation and no CUDA errors. Verify correctness in a separate startup process so it does not warm the timed server.

For the four warm responses, pool `/completion` timing fields: PP = `1000 * sum(timings.prompt_n) / sum(timings.prompt_ms)`; TG = `1000 * sum(timings.predicted_n) / sum(timings.predicted_ms)`. Warm prompt + decode seconds = `sum(prompt_ms + predicted_ms) / 1000`. Exclude the first response; do not average per-request rates. Here `predicted_n=512` counts returned generated tokens, whereas the routing diagnostic attributes 511 generated forwards per request: the first output comes from prefill. The resulting 2555 attributed forwards across all five requests are a cache-counter denominator, not the TG numerator.

For the internal three-step benchmark, the retained sampling above is intentional. The custom Pi fork and private benchmark inputs are not bundled here, so the configuration is documented but the historical benchmark cannot be reproduced from this repository alone.

## Measurement method

1. Record source revision plus dirty diff, binary/library hashes, GGUF/seed hashes, command and environment. Check startup allocation and basic correctness with Q8 KV/FA before accepting timing.
2. Separate cold first request, warm fixed work and live tasks. Keep model/quant/context/KV/FA/batch/ubatch/threads/sampling unchanged except the stated controlled parameter. A short native llama-bench run is not evidence of server-only periodic attribution.
3. Pool raw token counts and timing pairs rather than averaging rates. Use raw slot timings when harness summaries omit decode work.
4. Report counts, correctness, PP, TG, task/process duration and peak memory. Different live trajectories prevent pure implementation wall-time attribution. Record trial dispersion/order; do not infer statistical confidence from one live run.
5. Historical guards sampled GPU memory at 250 ms, vmstat at 1 s and process/RSS/I/O at 5 s; no concurrent inference/tests/builds. Preferred peak 7500 MiB, hard guard 7680 MiB, zero swap-counter delta. These are harness policies, not server-enforced limits. Historical WSL runs disabled then explicitly restored their identified swap device; do not blindly apply those host-specific commands elsewhere.

## Memory and correctness

At the optimized CPU35 profile: CUDA model 4426.80 MiB, KV/recurrent state 1205.00 MiB, CUDA compute 423.40 MiB, host compute 32.34 MiB, before hot-store/runtime additions. The S18 store reports roughly 1.19 GiB over 105 expert tensors. Complete measured peaks, rather than the sum of selected buffers, determine the usable margin.

Skipping 20 unused MTP tensors saves exactly 371.36 MiB for this GGUF. Compact masks remove the dense 430 MiB host prompt mask; CUDA allocator savings differ because lifetimes overlap. The recorded old-to-ported accounted CUDA reduction was 868.96 MiB at that allocation stage. Do not add host and device deltas or assume the same savings at all contexts/backends.

Recorded checks include prompt/decode arithmetic/code correctness, compact-mask CPU/CUDA equivalence, exact output parity in matched execution fixtures, pending-upload lifetime/abort/recovery tests, model-indexed persistence and task validations. These do not establish general perplexity equivalence or current-upstream long-context numerical equivalence.

Refreshed staging/MMQ test objects linked against the frozen repaired libraries passed all eight Node fixture suites, nine guarded MMQ cases and 14 compact-mask equivalence cases each on CPU and CUDA. The guarded MMQ tests require their historical `CUDA_LAUNCH_BLOCKING=1` setting: both the retained original test and refreshed test failed without it. Immediate guarded-pool unmapping while work remains asynchronous is a plausible fixture-lifetime explanation, not a proven production regression or a new correctness fix. Inference runs above pass with launch blocking off.

This is focused runtime evidence, not a fresh full production build or a pass of the expanded automatic CUDA CTest manifest. Guard-test environment integration and clean-checkout release validation remain open.

## Reproducibility limits

The tables summarize retained local measurements. Raw task logs, the task harness/corpus, the model-compatible warm seed and the frozen prompt token array are not bundled in this repository. Commands, counter definitions, dates, revisions, token counts and input checksums are documented above; an exact replay of the historical runs still requires those inputs.

The rounded headline comparison and approximately-90%-hit observation still need exact run provenance. Exact historical replay also requires a benchmark input package and recorded laptop power/fan settings.

Inherited RTX 3070/RX 570 author measurements are available in the [original wackMall README](https://github.com/miltos22/llama-wackMall/blob/39d867bb676a492ed6a667a133d643315ac20474/README.md); they are not this laptop's results.
