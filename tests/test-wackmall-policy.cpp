#include "arg.h"
#include "common.h"
#include "llama-expert-tier.h"

#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static std::vector<char *> argv_ptrs(std::vector<std::string> & argv) {
    std::vector<char *> result;
    for (std::string & arg : argv) {
        result.push_back(arg.data());
    }
    return result;
}

int main() {
    common_params params;
    std::vector<std::string> argv = {"test-wackmall-policy", "-m", "model.gguf", "--fit", "off"};
    std::vector<char *> args = argv_ptrs(argv);

    assert(common_params_parse(args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cmoe);
    assert(!params.fit_params);

    // A caller-provided S is authoritative; affordability only selects an automatic S.
    assert(llama_expert_tier::resolve_slot_count(37, true, 27, 64) == 37);
    assert(llama_expert_tier::resolve_slot_count(37, false, 27, 64) == 27);

    common_params placement_params;
    argv = {"test-wackmall-policy", "-m", "model.gguf", "--n-cpu-moe", "35"};
    args = argv_ptrs(argv);
    assert(common_params_parse(args.size(), args.data(), placement_params, LLAMA_EXAMPLE_COMMON));
    size_t placement_override_count = 0;
    while (placement_override_count < placement_params.tensor_buft_overrides.size() &&
           placement_params.tensor_buft_overrides[placement_override_count].pattern != nullptr) {
        ++placement_override_count;
    }
    assert(placement_override_count == 35);

    common_params runtime_params;
    argv = {
        "test-wackmall-policy", "-m", "model.gguf",
        "--parallel", "3", "--no-kv-unified", "--no-kv-offload", "--flash-attn", "off",
    };
    args = argv_ptrs(argv);
    assert(common_params_parse(args.size(), args.data(), runtime_params, LLAMA_EXAMPLE_SERVER));
    assert(runtime_params.n_parallel == 3);
    assert(!runtime_params.kv_unified);
    assert(runtime_params.no_kv_offload);
    assert(runtime_params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED);

    assert(llama_expert_tier::is_decode_batch_size(1, 16));
    assert(llama_expert_tier::is_decode_batch_size(16, 16));
    assert(!llama_expert_tier::is_decode_batch_size(17, 16));
    assert(!llama_expert_tier::is_decode_batch_size(2048, 16));
    return 0;
}
