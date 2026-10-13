# llama.cpp Hot Experts fork

This is a fork of [miltos22's wackMall](https://github.com/miltos22/llama-wackMall), intended primarily for agentic coding on my Ryzen 9 7940HS / RTX 4060 Laptop (32 GB RAM, 8 GB VRAM). It restores mixed CPU-MoE placement alongside wackMall's GPU-hot / CPU-cold execution, reclaims memory for hot experts and reduces CPU/GPU handoff overhead. Development and benchmarks focus on Tiel Coder, which I use daily.

## Why I picked wackMall

Layer-granular `--n-cpu-moe` placement leaves VRAM that cannot fit another complete expert layer. wackMall's resident GPU experts can use that remainder while misses execute on CPU, rather than uploading missing expert weights for GPU execution. That was the best architectural fit for this laptop among the approaches I tested. [Cache experiments](docs/BENCHMARKS.md#earlier-expert-cache-experiments) include the later upstream weight-upload cache comparison.

Original wackMall was not faster out of the box: its tier-enabled launch forced all expert layers onto CPU and batch/ubatch to 256. The warm S8 three-step run measured 116.60 PP / 23.32 TG; S35 reached 24.79 TG, still below the static CPU33 reference. [Historical results](docs/BENCHMARKS.md#from-original-wackmall-to-this-fork) record the configuration differences.

I restored configurable CPU-MoE placement to retain GPU-resident expert layers for prefill, instead of trading them all for cache capacity. The batch override also changed; the historical PP difference is not a placement-only measurement.

## What this fork changes

- **Restore mixed placement.** Respect `--n-cpu-moe` and requested batch/KV settings. On the tested 40-layer model, CPU35 retains five GPU expert layers and caches hot experts from the other 35.
- **Make room for the cache.** Skip unused MTP tensors and port compact causal Flash Attention masks. On the tested Tiel GGUF, the unused tensors alone cost 371.36 MiB; the old prompt mask also consumed hundreds of MiB.
- **Reduce handoff overhead.** Add protected input staging, fused CUDA submission and guarded asynchronous cold-result uploads, with completion and buffer-lifetime checks.
- **Bound adaptation and expose diagnostics.** Add optional generated-token-only periodic adaptation, isolated persistence paths and routing/transfer counters. Adaptation adds a small gain in the longer replay, not the main speedup.
- **Fix CUDA scratch allocation.** Repair an inherited MMQ scratch-tail allocation defect and add regression tests.

The GPU-hot / CPU-cold design, fused CPU cold operator and original residency mechanisms are wackMall's work. The placement, memory, execution and bounded-policy changes above are this fork's work. [CHANGES](docs/CHANGES.md) separates them; [HOT_EXPERTS](docs/HOT_EXPERTS.md) explains the implementation.

## Performance at a glance

PP1024/TG512, same repaired CUDA 12.4 server: Tiel Q4_K_XL, context 110000, Q8 K/V, eight threads. Warm rates pool four requests (4096 prompt / 2048 generated tokens), without prefix reuse. Baseline: CPU33/S0. [First-request rates and launch parameters](docs/BENCHMARKS.md#fresh-same-binary-pp1024tg512).

| Configuration | Warm PP tok/s | PP delta | Warm TG tok/s | TG delta | Peak VRAM MiB | VRAM delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU33, S=0 | 551.19 | 0.00% | 29.55 | 0.00% | 7,167 | 0 |
| CPU35, S18 optimized | 526.85 | -4.41% | 33.71 | +14.06% | 7,457 | +290 |

Summed warm prompt/decode time falls 10.68%. Placement, cache and execution settings change together; the diagnostic records zero adaptive replacements in this short run.

The secondary [internal three-step agentic benchmark on my custom Pi fork](docs/BENCHMARKS.md#internal-three-step-benchmark-results) measures 28.07 TG for CPU33/S0, 31.32 for fixed S18 (+11.56%) and 31.44 for adaptive S18 (+11.99%). All pass the three steps and both validations. Prompt lengths and agent trajectories differ, so its PP and task durations are not fixed-work speedups.

[Detailed results](docs/BENCHMARKS.md) include upstream cache comparisons, original wackMall measurements and the separate non-cache/cache attribution summary.

## Tested scope

Development centers on an ASUS ROG Zephyrus G14 (2023): Ryzen 9 7940HS, RTX 4060 Laptop 8 GB, 32 GB DDR5, Windows 11 + WSL2. The retained fork measurements use CUDA 12.4 and Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL (`qwen35moe`).

The daily-use model scope includes KAT-Coder-V2.5-Dev, Tiel Coder and Qwen 3.6 35B-A3B-class coding models. The published numerical comparisons here are Tiel Q4_K_XL, not measurements of every model/quantization. Other hardware and architectures are not systematically tested; that does not imply incompatibility.

## Build and run

From the repository root, with CMake, a C++ compiler and CUDA 12.4 installed:

```bash
export CUDA124_ROOT=/path/to/cuda-12.4
cmake -S . -B build-cuda124-sm89 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA124_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA124_ROOT" -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda124-sm89 -j 8 --target llama-server
```

Minimal fixed-cache server example for the tested Tiel model, in a clean shell:

```bash
export MODEL=/path/to/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
export HOT_SEED=/path/to/model-compatible-heat.csv
LLAMA_EXPERT_S=18 LLAMA_EXPERT_HOT="$HOT_SEED" LLAMA_EXPERT_ADAPT=0 \
LLAMA_EXPERT_SIDECAR_IN= LLAMA_EXPERT_SIDECAR_OUT= \
./build-cuda124-sm89/bin/llama-server \
  -m "$MODEL" --host 127.0.0.1 --port 8080 --jinja --no-mmproj \
  -c 110000 -ngl all --n-cpu-moe 35 -b 2048 -ub 2048 -np 1 \
  -ctk q8_0 -ctv q8_0 --flash-attn on -fit off -t 8
```

The seed must match the model; `tests/wackmall-staging-seed.csv` is only a synthetic test fixture. This example enables fixed residency, not the complete optimized benchmark profile. See [benchmark reproduction](docs/BENCHMARKS.md#reproducing-the-tested-profile) for that profile, controls and validation. For other build backends, see the inherited [build guide](docs/build.md).

## Important controls

| Setting | Purpose |
| --- | --- |
| `--n-cpu-moe N` | Keep expert weights of the first N layers on CPU; larger N leaves more potential cache room. |
| `LLAMA_EXPERT_S` | Hot slots per eligible layer, not server slots. `0` disables the hot tier; unset selects an automatic budget. |
| `LLAMA_EXPERT_HOT` | Warm-start heat CSV; takes precedence over sidecar input. |
| `LLAMA_EXPERT_ADAPT` / `LLAMA_EXPERT_PERIODIC` | Adaptation and opt-in bounded generated-decode policy. |
| `LLAMA_EXPERT_STAGE_INPUTS`, `LLAMA_EXPERT_FUSE_STAGING`, `LLAMA_EXPERT_ASYNC_COLD_UPLOAD` | Opt-in execution paths used together in the measured optimized profile. |
| `LLAMA_EXPERT_SIDECAR_IN` / `OUT` | Independent persistence paths; empty disables that direction. |

Defaults are not the benchmark profile. In particular, bounded periodic mode requires inherited pool/pread/prefetch/prediction/page-hint features off. [Architecture and configuration](docs/HOT_EXPERTS.md#configuration) explains the controls and fallbacks; [benchmarks](docs/BENCHMARKS.md) records measured tradeoffs and earlier experiments.

## Credits

[llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) provide the runtime, operators, backends and tooling. [miltos22's wackMall](https://github.com/miltos22/llama-wackMall) provides the resident hot-expert / CPU cold-expert foundation and inherited adaptation/pooling mechanisms. This fork's work is identified separately in [CHANGES.md](docs/CHANGES.md); compact causal masks were ported using newer llama.cpp behavior as a specification.

Original licenses and notices are retained: [LICENSE](LICENSE) and [NOTICE](NOTICE). The inherited [architecture disclosure](ARCHITECTURE.md) remains available as a historical record.
