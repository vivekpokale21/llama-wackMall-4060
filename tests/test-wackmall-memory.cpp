#include "common.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-model.h"

#undef NDEBUG
#include <cassert>

struct empty_model : llama_model_base {
    empty_model() : llama_model_base(llama_model_default_params()) { arch = LLM_ARCH_QWEN35MOE; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return nullptr; }
};

int main() {
    common_params params;
    assert(!common_model_params_to_llama(params).load_mtp);
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    assert(common_model_params_to_llama(params).load_mtp);
    assert(!llama_model_default_params().load_mtp);

    empty_model model;
    llama_kv_cache cache(model, model.hparams, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                         false, false, false, 256, 1, 256, 0, LLAMA_SWA_TYPE_NONE,
                         nullptr, {}, {}, {});
    llama_pos positions[] = {0, 1, 2};
    int32_t counts[] = {1, 1, 1};
    llama_seq_id seq = 0;
    llama_seq_id * ids[] = {&seq, &seq, &seq};
    llama_ubatch batch {};
    batch.n_tokens = batch.n_seq_tokens = 3;
    batch.n_seqs = batch.n_seqs_unq = batch.n_pos = 1;
    batch.b_equal_seqs = 1;
    batch.pos = positions;
    batch.n_seq_id = counts;
    batch.seq_id = ids;
    batch.seq_id_unq = &seq;
    assert(cache.can_use_compact_causal_mask(batch, true, 256, true));
    assert(!cache.can_use_compact_causal_mask(batch, false, 256, true));
    assert(!cache.can_use_compact_causal_mask(batch, true, 256, false));
    auto slot = cache.find_slot(batch, true);
    assert(!slot.empty());
    cache.apply_ubatch(slot, batch);
    assert(cache.can_use_compact_causal_mask(batch, true, 256, false, &slot));
    positions[1] = 3;
    assert(!cache.can_use_compact_causal_mask(batch, true, 256, false, &slot));
    positions[1] = 1;
    auto reordered = slot;
    std::swap(reordered.idxs[0][0], reordered.idxs[0][1]);
    assert(!cache.can_use_compact_causal_mask(batch, true, 256, false, &reordered));
    batch.n_seqs_unq = 2;
    assert(!cache.can_use_compact_causal_mask(batch, true, 256, false, &slot));
    batch.n_seqs_unq = 1;
    model.hparams.use_alibi = true;
    assert(!cache.can_use_compact_causal_mask(batch, true, 256, false, &slot));
    model.hparams.use_alibi = false;
    assert(cache.seq_rm(0, 1, 2));
    assert(!cache.can_use_compact_causal_mask(batch, true, 256, false, &slot));
    return 0;
}
