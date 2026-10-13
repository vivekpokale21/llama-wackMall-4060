#include "llama-batch.h"
#include "llama-expert-tier.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "wackmall-async-reader.h"
#include "wackmall-cold-upload-probe.h"
#include "wackmall-test-env.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

struct staging_model : llama_model_base {
    staging_model() : llama_model_base(llama_model_default_params()) {
        arch = LLM_ARCH_QWEN35MOE;
        hparams = {};
        hparams.n_embd = 32;
        hparams.n_layer_all = 3;
        hparams.n_expert = 8;
        hparams.n_expert_used = 2;
        hparams.n_head_arr.fill(1);
        hparams.n_head_kv_arr.fill(1);
        hparams.n_embd_head_k_full = hparams.n_embd_head_v_full = 32;
        layers.resize(3);
    }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return nullptr; }
};

static std::vector<float> run(staging_model & model, ggml_backend_t gpu, ggml_backend_t cpu, int tokens, bool staging, bool tier_active) {
    if (staging) {
        setenv("LLAMA_EXPERT_STAGE_INPUTS", "1", 1);
    } else {
        unsetenv("LLAMA_EXPERT_STAGE_INPUTS");
    }
    ggml_backend_t backends[] = {gpu, cpu};
    const char * fuse = getenv("LLAMA_EXPERT_FUSE_STAGING");
    ggml_backend_buffer_type_t bufts[] = {ggml_backend_get_default_buffer_type(gpu),
        fuse && atoi(fuse) ? ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu)) : ggml_backend_get_default_buffer_type(cpu)};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 2048, false, true);
    if (getenv("WACKMALL_TEST_CALLBACK")) {
        ggml_backend_sched_set_eval_callback(sched, [](ggml_tensor *, bool, void *) { return false; }, nullptr);
    }
    llm_graph_result result(2048);
    llama_adapter_loras loras;
    llm_graph_params params {};
    params.arch = model.arch;
    params.hparams = model.hparams;
    params.ubatch.n_tokens = tokens;
    params.sched = sched;
    params.backend_cpu = cpu;
    params.loras = &loras;
    params.res = &result;
    params.cb = [](const llama_ubatch &, ggml_tensor * t, const char * name, int il) {
        ggml_format_name(t, "%s-%d", name, il);
    };
    llm_graph_context builder(params);
    ggml_context * ctx = result.get_ctx();
    ggml_cgraph * graph = result.get_gf();
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, tokens);
    ggml_set_input(input);
    // The reuse/abort probes resubmit without rewriting inputs; forbid in-place
    // scale reuse so those probes retain the same input bytes on both submits.
    if (getenv("WACKMALL_TEST_COLD_UPLOAD_GUARD")) ggml_set_output(input);
    ggml_backend_sched_set_tensor_backend(sched, input, gpu);
    ggml_tensor * cur = ggml_scale(ctx, input, 0.75f);
    ggml_backend_sched_set_tensor_backend(sched, cur, gpu);
    ggml_tensor * join_input = nullptr;
    if (getenv("WACKMALL_TEST_JOIN_INPUT") && staging && tokens == 1) {
        join_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, tokens);
        ggml_set_input(join_input);
        ggml_set_output(join_input);
        ggml_set_name(join_input, "join_user_input");
        ggml_backend_sched_set_tensor_backend(sched, join_input, cpu);
    }
    for (int il = 0; il < 3; ++il) {
        const auto & layer = model.layers[il];
        cur = builder.build_moe_ffn(cur, layer.ffn_gate_inp, layer.ffn_up_exps,
                layer.ffn_gate_exps, layer.ffn_down_exps, nullptr, 8, 2,
                LLM_FFN_SILU, true, 1.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
        if (join_input) cur = ggml_add(ctx, cur, join_input);
    }
    ggml_set_output(cur);
    ggml_build_forward_expand(graph, cur);
    std::vector<ggml_tensor *> staged;
    std::vector<ggml_tensor *> cold_nodes;
    struct hot_store {
        ggml_tensor * hot;
        ggml_tensor * lut;
        ggml_tensor * source;
    };
    std::vector<hot_store> hot_stores;
    int cold_count = 0;
    int last_stage = -1;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        if (strncmp(t->name, "moe_stage_", 10) == 0) {
            assert(t->op == GGML_OP_CPY);
            assert(t->flags & GGML_TENSOR_FLAG_OUTPUT);
            assert(t->view_src && (t->view_src->flags & GGML_TENSOR_FLAG_OUTPUT));
            assert(ggml_backend_sched_get_tensor_backend(sched, t) == cpu);
            staged.push_back(t);
            last_stage = i;
        }
        if (t->op == GGML_OP_MUL_MAT_ID && strstr(t->src[0]->name, ".hot")) {
            auto * rows = t->src[2]->view_src;
            assert(rows && rows->op == GGML_OP_GET_ROWS);
            ggml_tensor * source = nullptr;
            for (const auto & layer : model.layers) {
                for (auto * w : {layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
                    if (std::string(w->name) + ".hot" == t->src[0]->name) {
                        source = w;
                    }
                }
            }
            assert(source);
            hot_stores.push_back({t->src[0], rows->src[0], source});
            if (staging && tier_active && tokens == 1) {
                assert(last_stage >= 0 && last_stage < i);
            }
        }
        if (t->op == GGML_OP_MOE_COLD) {
            ++cold_count;
            cold_nodes.push_back(t);
            if (staging && tier_active && tokens == 1) {
                assert(strncmp(t->src[3]->name, "moe_stage_", 10) == 0);
                assert(strncmp(t->src[4]->name, "moe_stage_", 10) == 0);
                if (getenv("WACKMALL_TEST_COLD_UPLOAD_GUARD")) assert(t->flags & GGML_TENSOR_FLAG_OUTPUT);
            }
        }
    }
    assert(cold_count == (tier_active && tokens <= 16 ? 2 : 0));
    assert(staged.size() == (staging && tier_active && tokens == 1 ? 4 : 0));
    assert(ggml_backend_sched_alloc_graph(sched, graph));
    for (ggml_tensor * t : staged) {
        assert(ggml_backend_sched_get_tensor_backend(sched, t) == cpu);
    }
    for (size_t i = 0; i < staged.size(); ++i) {
        assert(ggml_backend_buffer_is_host(staged[i]->buffer));
        for (size_t j = 0; j < i; ++j) {
            const uintptr_t a = (uintptr_t) staged[i]->data;
            const uintptr_t b = (uintptr_t) staged[j]->data;
            assert(a + ggml_nbytes(staged[i]) <= b || b + ggml_nbytes(staged[j]) <= a);
        }
    }
    if (getenv("WACKMALL_TEST_COLD_UPLOAD_GUARD") && staging && tier_active && tokens == 1) {
        for (size_t i = 0; i < cold_nodes.size(); ++i) {
            for (size_t j = 0; j < i; ++j) {
                const uintptr_t a = (uintptr_t) cold_nodes[i]->data, b = (uintptr_t) cold_nodes[j]->data;
                assert(a + ggml_nbytes(cold_nodes[i]) <= b || b + ggml_nbytes(cold_nodes[j]) <= a);
            }
        }
    }
    std::vector<float> values(32*tokens);
    std::vector<float> output;
    const bool periodic = getenv("WACKMALL_TEST_PERIODIC") != nullptr;
    const bool no_adapt = getenv("WACKMALL_TEST_PERIODIC_NO_ADAPT") != nullptr;
    const bool unattributed = getenv("WACKMALL_TEST_PERIODIC_UNATTRIBUTED") != nullptr;
    const char * reference_phase = getenv("WACKMALL_TEST_REFERENCE_PHASE");
    const bool asynchronous = getenv("WACKMALL_TEST_ASYNC") != nullptr;
    if (asynchronous) {
        output.reserve((periodic ? 2147 : 8) * ggml_nelements(cur));
    }
    // Reuse the allocated graph with changing input; all CPU/GPU users must complete before reuse.
    for (int step = 0; step < (periodic ? 2147 : 8); ++step) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.2f + std::sin(float(i + step*3))*0.05f;
        }
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        if (join_input) {
            const std::vector<float> zeros(32 * tokens, 0.0f);
            ggml_backend_tensor_set(join_input, zeros.data(), 0, ggml_nbytes(join_input));
        }
        const char * generated = getenv("WACKMALL_TEST_TIMING_GENERATED");
        ggml_backend_sched_set_generated_decode(sched, generated && atoi(generated) != 0 && tokens == 1);
        const char * guard_mode = getenv("WACKMALL_TEST_COLD_UPLOAD_GUARD");
        const bool guard = guard_mode && staging && tier_active && tokens == 1 && step == 7;
        std::unique_ptr<wackmall_cold_upload_probe> upload;
        void * final_output = nullptr;
        if (guard && strcmp(guard_mode, "free") == 0) {
            assert(cudaMallocHost(&final_output, ggml_nbytes(cur)) == cudaSuccess);
        }
        const bool abort_upload = guard && strcmp(guard_mode, "abort") == 0;
        if (guard) upload.reset(new wackmall_cold_upload_probe(gpu, cold_nodes.back(), abort_upload));
        assert((asynchronous || guard ? ggml_backend_sched_graph_compute_async(sched, graph) :
            ggml_backend_sched_graph_compute(sched, graph)) == (abort_upload ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS));
        if (guard) {
            upload->pending();
            if (strcmp(guard_mode, "reuse") == 0) assert(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
            if (strcmp(guard_mode, "reset") == 0) ggml_backend_sched_reset(sched);
            if (strcmp(guard_mode, "abort") == 0) {
                ggml_backend_sched_synchronize(sched);
                assert(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
            }
            if (strcmp(guard_mode, "free") == 0) {
                const size_t offset = output.size();
                output.resize(offset + ggml_nelements(cur));
                // A pageable D2H destination can block submission before the
                // free guard, defeating the intentionally pending DMA probe.
                ggml_backend_tensor_get_async(gpu, cur, final_output, 0, ggml_nbytes(cur));
                ggml_backend_sched_free(sched);
                sched = nullptr;
                upload.reset();
                memcpy(output.data() + offset, final_output, ggml_nbytes(cur));
                assert(cudaFreeHost(final_output) == cudaSuccess);
                fprintf(stderr, "PASS: pending cold upload drained at %s; source and uploaded bytes preserved\n", guard_mode);
                break;
            }
        }
        for (size_t il = 0; il < cold_nodes.size(); ++il) {
            ggml_tensor * cold = cold_nodes[il];
            const int32_t * ids = (const int32_t *) cold->src[4]->data;
            assert(ids[0] == 0 && ids[1] == 2);
            const int32_t * mask = (const int32_t *) cold->src[5]->data;
            const bool cold_two = reference_phase ? (int) il >= atoi(reference_phase) :
                !periodic || no_adapt || unattributed || step < 32 + 2048 + (int) il * 32;
            assert(mask[0] == 0 && mask[2] == (cold_two ? 1 : 0));
        }
        const size_t offset = output.size();
        output.resize(offset + ggml_nelements(cur));
        if (asynchronous) {
            ggml_backend_tensor_get_async(ggml_backend_sched_get_tensor_backend(sched, cur), cur,
                output.data() + offset, 0, ggml_nbytes(cur));
        }
        const bool probe = asynchronous && periodic && !hot_stores.empty() &&
            (step == 32 || step == 63 || step == 2079 || step == 2111 || step == 2146);
        std::unique_ptr<wackmall_async_reader> reader;
        if (probe) {
            std::vector<ggml_tensor *> tensors;
            for (const auto & store : hot_stores) {
                tensors.push_back(store.hot);
                tensors.push_back(store.lut);
            }
            reader.reset(new wackmall_async_reader(gpu, tensors));
        }
        llama_expert_tier::synchronize_and_update(sched, tokens, periodic && !unattributed && step >= 32);
        if (upload) {
            upload.reset();
            fprintf(stderr, "PASS: pending cold upload drained at %s; source and uploaded bytes preserved\n", guard_mode);
        }
        reader.reset();
        if (probe) {
            for (const auto & store : hot_stores) {
                std::vector<int32_t> lut(8);
                std::vector<char> hot(ggml_nbytes(store.hot));
                ggml_backend_tensor_get(store.lut, lut.data(), 0, lut.size() * sizeof(int32_t));
                ggml_backend_tensor_get(store.hot, hot.data(), 0, hot.size());
                const size_t slice = ggml_nbytes(store.source) / 8;
                int layer_id = -1;
                assert(sscanf(store.source->name, "blk.%d.", &layer_id) == 1);
                for (int expert = 0; expert < 8; ++expert) {
                    assert(lut[expert] >= 0 && lut[expert] <= 2);
                    for (auto * cold : cold_nodes) {
                        int cold_layer = -1;
                        assert(sscanf(cold->src[0]->name, "blk.%d.", &cold_layer) == 1);
                        if (cold_layer == layer_id) {
                            assert(((int32_t *) cold->src[5]->data)[expert] == (lut[expert] == 2));
                        }
                    }
                    if (lut[expert] < 2) {
                        assert(memcmp(hot.data() + lut[expert] * slice,
                            (char *) store.source->data + expert * slice, slice) == 0);
                    }
                }
            }
            fprintf(stderr, "async_reader: step %d old GPU weights/LUTs preserved through wait; published weights match GPU LUT\n", step);
        }
        if (!asynchronous) {
            ggml_backend_tensor_get(cur, output.data() + offset, 0, ggml_nbytes(cur));
        }
        for (float value : output) {
            assert(std::isfinite(value));
        }
    }
    ggml_backend_sched_free(sched);
    if (periodic) {
        const char * path = getenv("WACKMALL_TEST_OUTPUT");
        assert(path);
        FILE * f = fopen(path, "wb");
        assert(f && fwrite(output.data(), sizeof(float), output.size(), f) == output.size());
        assert(fclose(f) == 0);
    }
    return output;
}

int main(int argc, char ** argv) {
    wackmall_test_output_env();
    assert(argc == 2 || argc == 3);
    const bool tier_active = argc == 2;
    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(gpu && cpu);
    staging_model model;
    ggml_context * ctx_cpu = ggml_init({1024*1024, nullptr, true});
    ggml_context * ctx_gpu = ggml_init({1024*1024, nullptr, true});
    for (int il = 0; il < 3; ++il) {
        auto & layer = model.layers[il];
        const char * gpu_layer = getenv("WACKMALL_TEST_GPU_LAYER");
        ggml_context * ctx = il == (gpu_layer ? atoi(gpu_layer) : 2) ? ctx_gpu : ctx_cpu;
        layer.ffn_gate_inp = ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_F32, 32, 8);
        layer.ffn_gate_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, 8);
        layer.ffn_up_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, 8);
        layer.ffn_down_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, 8);
        ggml_format_name(layer.ffn_gate_exps, "blk.%d.ffn_gate_exps", il);
        ggml_format_name(layer.ffn_up_exps, "blk.%d.ffn_up_exps", il);
        ggml_format_name(layer.ffn_down_exps, "blk.%d.ffn_down_exps", il);
    }
    ggml_backend_buffer_t host = ggml_backend_alloc_ctx_tensors(ctx_cpu, cpu);
    ggml_backend_buffer_t device = ggml_backend_alloc_ctx_tensors(ctx_gpu, gpu);
    assert(host && device);
    ggml_backend_buffer_set_usage(host, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(device, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto & layer : model.layers) {
        for (ggml_tensor * t : {layer.ffn_gate_inp, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
            std::vector<float> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); ++i) {
                if (t == layer.ffn_gate_inp) {
                    const int expert = i / 32;
                    data[i] = expert == 0 ? 0.1f : expert == 2 ? 0.08f : -0.1f;
                } else {
                    data[i] = 0.08f + std::cos(float(i)*0.37f)*0.01f;
                }
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
    }
    setenv("LLAMA_EXPERT_S", tier_active ? "2" : "0", 1);
    if (getenv("WACKMALL_TEST_SIDECAR_LOAD")) {
        unsetenv("LLAMA_EXPERT_HOT");
    } else {
        setenv("LLAMA_EXPERT_HOT", argv[1], 1);
    }
    setenv("LLAMA_EXPERT_ADAPT", getenv("WACKMALL_TEST_PERIODIC") && !getenv("WACKMALL_TEST_PERIODIC_NO_ADAPT") ? "1" : "0", 1);
    setenv("LLAMA_EXPERT_MADVISE", "0", 1);
    setenv("LLAMA_EXPERT_RAMPOOL", "0", 1);
    setenv("LLAMA_EXPERT_PREAD", "0", 1);
    setenv("LLAMA_EXPERT_PREDICT", "0", 1);
    setenv("LLAMA_EXPERT_PREFETCH_GB", "0", 1);
    setenv("LLAMA_EXPERT_TMAX", "16", 1);
    llama_expert_tier::init(model);
    if (getenv("WACKMALL_TEST_PERIODIC")) {
        run(model, gpu, cpu, 1, true, tier_active);
    } else for (int tokens : {1, 4, 17}) {
        const auto baseline = run(model, gpu, cpu, tokens, false, tier_active);
        const auto experiment = run(model, gpu, cpu, tokens, true, tier_active);
        assert(baseline.size() == experiment.size());
        for (size_t i = 0; i < baseline.size(); ++i) {
            assert(std::fabs(baseline[i] - experiment[i]) <= 1e-6f);
        }
    }
    ggml_backend_buffer_free(host);
    ggml_backend_buffer_free(device);
    ggml_free(ctx_cpu);
    ggml_free(ctx_gpu);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    puts("PASS: one-token staging order, protected host regions, graph reuse and bypass parity");
}
