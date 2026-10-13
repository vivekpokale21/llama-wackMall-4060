#include "llama-expert-policy.h"
#include "llama-expert-tier.h"
#undef NDEBUG
#include <cassert>

int main() {
    using namespace llama_expert_tier;
    assert(server_generated_decode(1, 1, false, true));
    assert(!server_generated_decode(1, 1, false, false));
    assert(!server_generated_decode(1, 1, true, true));
    assert(!server_generated_decode(2, 1, false, true));
    assert(!server_generated_decode(1, 2, false, true));
    llama_token token = 42;
    llama_seq_id seq = 0;
    llama_seq_id * seqs[] = { &seq };
    int32_t n_seq = 1;
    llama_batch batch {};
    batch.n_tokens = 1;
    batch.token = &token;
    batch.n_seq_id = &n_seq;
    batch.seq_id = seqs;
    assert(generated_batch_shape(batch, 1));
    assert(!generated_batch_shape(batch, 2));
    batch.n_tokens = 2;
    assert(!generated_batch_shape(batch, 1));
    batch.n_tokens = 1;
    float embedding = 0.0f;
    batch.embd = &embedding;
    assert(!generated_batch_shape(batch, 1));
    batch.embd = nullptr;
    n_seq = 2;
    assert(!generated_batch_shape(batch, 1));
    n_seq = 1;
    seq = 1;
    assert(!generated_batch_shape(batch, 1));
    seq = 0;
    batch.seq_id = nullptr;
    assert(!generated_batch_shape(batch, 1));
    return 0;
}
