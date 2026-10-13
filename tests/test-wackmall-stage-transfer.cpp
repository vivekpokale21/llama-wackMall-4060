#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>

#undef NDEBUG
#include <cassert>

struct timing_gate { std::atomic<bool> release {false}, finished {false}, timed_out {false}; };
static void CUDART_CB hold_join(void * data) {
    auto & gate = *(timing_gate *) data;
    while (!gate.release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    assert(gpu);
    ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto fused = (ggml_backend_fused_stage_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fused_stage");
    assert(fused);
    using fused_timing = ggml_backend_fused_timing;
    using timing_fn = ggml_backend_fused_timing_t;
    auto timing = (timing_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fused_timing");
    const bool test_timing = getenv("WACKMALL_TEST_FUSED_TIMING") != nullptr;
    if (test_timing) {
        assert(timing && "missing opt-in fused suffix timing extension");
        timing(gpu, GGML_BACKEND_FUSED_TIMING_ARM, nullptr);
        timing(gpu, GGML_BACKEND_FUSED_TIMING_CANCEL, nullptr);
        fused_timing cancelled {};
        timing(gpu, GGML_BACKEND_FUSED_TIMING_DETACH, &cancelled);
        assert(!cancelled.ready && cancelled.hot_us == 0.0);
    }
    for (int reset = 0; reset < 3; ++reset) {
        ggml_context * ctx = ggml_init({1024*1024, nullptr, true});
        ggml_context * host_ctx = ggml_init({1024*1024, nullptr, true});
        ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        ggml_tensor * x = ggml_scale(ctx, input, 0.5f);
        ggml_tensor * ids = ggml_argsort(ctx, input, GGML_SORT_ORDER_DESC);
        ggml_tensor * output = ggml_scale(ctx, x, 3.0f + reset);
        ggml_cgraph * full = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(full, x);
        ggml_build_forward_expand(full, ids);
        const int boundary = full->n_nodes;
        ggml_build_forward_expand(full, output);
        ggml_cgraph prefix = ggml_graph_view(full, 0, boundary);
        ggml_cgraph suffix = ggml_graph_view(full, boundary, full->n_nodes);
        ggml_tensor * x_dst = ggml_dup_tensor(host_ctx, x);
        ggml_tensor * ids_dst = ggml_dup_tensor(host_ctx, ids);
        ggml_tensor * x_copy = ggml_cpy(host_ctx, x, x_dst);
        ggml_tensor * ids_copy = ggml_cpy(host_ctx, ids, ids_dst);
        for (auto * t : {x_dst, ids_dst, x_copy, ids_copy}) ggml_set_output(t);
        ggml_backend_buffer_t device = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        ggml_backend_buffer_t host = ggml_backend_alloc_ctx_tensors_from_buft(host_ctx, ggml_backend_dev_host_buffer_type(dev));
        assert(device && host);
        const ggml_backend_stage_copy copies[] = {{x, x_copy}, {ids, ids_copy}};
        std::vector<float> values(32), result(32);
        for (int generation = 0; generation < 16; ++generation) {
            for (int i = 0; i < 32; ++i) values[i] = float((generation % 2 ? 31-i : i) + generation*32);
            ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
            const bool timed = test_timing && generation % 4 != 0;
            if (timed) timing(gpu, GGML_BACKEND_FUSED_TIMING_ARM, nullptr);
            int64_t wait = -1;
            assert(fused(gpu, &prefix, &suffix, copies, 2, &wait));
            assert(wait >= 0);
            const float * staged_x = (const float *) x_copy->data;
            const int32_t * staged_ids = (const int32_t *) ids_copy->data;
            for (int i = 0; i < 32; ++i) {
                assert(staged_x[i] == values[i]*0.5f);
                assert(staged_ids[i] == (generation % 2 ? i : 31-i));
            }
            timing_gate gate;
            cudaStream_t gate_stream = nullptr;
            ggml_backend_event_t gate_event = nullptr;
            std::thread watchdog;
            const bool blocked_join = timed && generation == 2;
            if (blocked_join) {
                assert(cudaStreamCreateWithFlags(&gate_stream, cudaStreamNonBlocking) == cudaSuccess);
                gate_event = ggml_backend_event_new(dev);
                assert(gate_event);
                watchdog = std::thread([&] {
                    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                    while (!gate.finished && std::chrono::steady_clock::now() < end) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    if (!gate.finished) { gate.timed_out = true; gate.release = true; }
                });
                assert(cudaLaunchHostFunc(gate_stream, hold_join, &gate) == cudaSuccess);
                assert(cudaEventRecord((cudaEvent_t) gate_event->context, gate_stream) == cudaSuccess);
                ggml_backend_event_wait(gpu, gate_event);
            }
            fused_timing t {};
            if (test_timing) timing(gpu, GGML_BACKEND_FUSED_TIMING_DETACH, &t);
            if (blocked_join) {
                void * pending = t.pending;
                assert(pending);
                timing(gpu, GGML_BACKEND_FUSED_TIMING_QUERY, &t);
                assert(!t.ready && t.pending == pending && !gate.timed_out);
                gate.release = true;
            }
            ggml_backend_synchronize(gpu);
            if (blocked_join) {
                gate.finished = true;
                watchdog.join();
                assert(!gate.timed_out);
                ggml_backend_event_free(gate_event);
                assert(cudaStreamDestroy(gate_stream) == cudaSuccess);
            }
            if (test_timing) {
                void * pending = t.pending;
                timing(gpu, GGML_BACKEND_FUSED_TIMING_QUERY, &t);
                if (t.ready != timed || t.pending != pending) {
                    fprintf(stderr, "query mismatch: generation=%d timed=%d ready=%d ownership=%d\n", generation, timed, t.ready, t.pending == pending);
                }
                assert(t.ready == timed && t.pending == pending);
                timing(gpu, GGML_BACKEND_FUSED_TIMING_COLLECT, &t);
                assert(t.ready == timed);
                assert(t.hot_us >= 0.0 && std::isfinite(t.hot_us));
                assert(t.hot_to_join_us >= 0.0 && std::isfinite(t.hot_to_join_us));
                assert(!t.pending);
            }
            ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
            for (int i = 0; i < 32; ++i) assert(result[i] == values[i]*(3.0f + reset)*0.5f);
        }
        auto malformed = copies[0];
        malformed.src = ids;
        ggml_backend_stage_copy unsupported[] = {malformed, copies[1]};
        assert(!fused(gpu, &prefix, &suffix, unsupported, 2, nullptr));
        ggml_tensor strided = *x;
        strided.nb[0] *= 2;
        unsupported[0] = {&strided, x_copy};
        assert(!fused(gpu, &prefix, &suffix, unsupported, 2, nullptr));
        x_copy->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
        assert(!fused(gpu, &prefix, &suffix, copies, 2, nullptr));
        ggml_backend_buffer_free(host);
        ggml_backend_buffer_free(device);
        ggml_free(host_ctx);
        ggml_free(ctx);
    }
    ggml_backend_free(gpu);
    puts("PASS: F32/I32 readiness, 48 generations, reallocation and eligibility rejection");
}
