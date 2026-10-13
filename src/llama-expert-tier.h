#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>

struct llama_model;

// expert hot-tiering: pins the most-used routed experts of each MoE layer in
// GPU memory; the remaining (cold) experts are computed on the CPU.
//
// enabled via env:
//   LLAMA_EXPERT_HOT   - path to heat csv with header "layer,expert,count"
//                        (seed; optional if LLAMA_EXPERT_ADAPT=1)
//   LLAMA_EXPERT_SIDECAR_IN/OUT - independent paths; unset: model.tier, empty: disabled
//   LLAMA_EXPERT_S     - hot slots per layer (default 16)
//   LLAMA_EXPERT_TMAX  - max n_tokens for the hot/cold path (default 16)
//   LLAMA_EXPERT_STATS - dump cold-hit stats at exit ("1" = stderr, else path)
//   LLAMA_EXPERT_ADAPT - 1: online repin of hot slots (decay + hysteresis)
//   LLAMA_EXPERT_PERIODIC - opt-in 32-token / 64-opportunity / one-global-promotion policy
//   LLAMA_EXPERT_OUTPUTS_BEFORE_SYNC - opt-in generated-decode output copies before the periodic tier wait
//   LLAMA_EXPERT_USAGE - dump learned hot set at exit (heat csv format)
//   LLAMA_EXPERT_OBSERVE - optional post-completion per-layer count JSONL

namespace llama_expert_tier {

inline bool generated_batch_shape(const llama_batch & batch, uint32_t n_seq_max) {
    return n_seq_max == 1 && batch.n_tokens == 1 && batch.token && !batch.embd &&
        batch.n_seq_id && batch.n_seq_id[0] == 1 && batch.seq_id && batch.seq_id[0] &&
        batch.seq_id[0][0] == 0;
}

// Explicit slot counts are caller policy. Affordability is only used when S is automatic.
inline int resolve_slot_count(int requested, bool manual, int affordable, int n_expert) {
    if (n_expert <= 0) {
        return 0;
    }
    const int selected = manual ? requested : affordable;
    return std::clamp(selected, 0, n_expert);
}

inline bool is_decode_batch_size(int64_t n_tokens, int tmax) {
    return n_tokens > 0 && n_tokens <= tmax;
}

// call once after the model tensors are loaded
void init(const llama_model & model);

// drop-in replacement for ggml_mul_mat_id on MoE expert weights:
// hot part on the GPU via the pinned store, cold part on the CPU via
// ggml_mul_mat_id_cold; falls back to plain ggml_mul_mat_id when the weight
// has no store or the batch is larger than LLAMA_EXPERT_TMAX
ggml_tensor * build_mul_mat_id(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids);

// True only when an initialized tier participates in this decode-sized graph.
bool active_for_tokens(int64_t n_tokens);

bool outputs_before_sync(int64_t n_tokens, bool generated_decode);

// Synchronize the tiered graph before consuming counts or mutating tier buffers.
// With LLAMA_EXPERT_STATS, time the existing wait and update separately.
// Token metadata is graph shape, not a prompt/decode classification.
// Periodic mode requires explicit generated-decode provenance, never graph-size inference.
void synchronize_and_update(ggml_backend_sched_t sched, int64_t n_tokens = 0, bool generated_decode = false);

// total bytes of routed-expert weight tensors (for dense-fit estimates)
LLAMA_API size_t expert_weight_bytes(const llama_model & model);

// fused cold-expert path for one MoE layer, two calls per layer:
// begin (before the expert matmuls): validates the layer and makes
//   build_mul_mat_id return hot-only results.
//   eligible must be false unless the layer uses separate gate/up tensors,
//   plain silu swiglu, and no expert biases/scales.
// end (after the down matmul, late in node order so the CPU cold op overlaps
//   GPU hot work): returns the cold contribution [embd, n_used, n_tokens]
//   (add it to the down result) or nullptr. x must be the layer input
//   captured before the matmuls.
bool begin_moe_cold(bool eligible,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * ids);

ggml_tensor * end_moe_cold(ggml_context * ctx,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * x, ggml_tensor * ids);

}
