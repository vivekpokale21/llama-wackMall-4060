#include "arg.h"
#include "common.h"
#include "llama-context.h"
#include "ggml-backend-impl.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>

#undef NDEBUG
#include <cassert>

// Instrument the real backend, including an outstanding CUDA tail at the tier wait.
static ggml_backend_i original;
static ggml_backend_event_t fence;
static cudaStream_t gate_stream;
static std::atomic<bool> release_gate;
static bool armed, submitted, copied, saw_wait, pending_at_wait;
static bool copy_before_wait;
static float * output_data;
static size_t output_count;
static int sync_count;
static ggml_backend_dispatch_profile_t profile;
static int reuses, launches, direct;

static void CUDART_CB hold_tail(void *) {
    while (!release_gate.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

static ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    profile(backend, nullptr, true);
    const auto status = original.graph_compute(backend, graph);
    ggml_backend_dispatch_profile dispatch {};
    profile(backend, &dispatch, false);
    reuses += dispatch.reuse;
    launches += dispatch.launches;
    direct += dispatch.direct;
    if (armed && status == GGML_STATUS_SUCCESS) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (strcmp(ggml_graph_node(graph, i)->name, "result_output") == 0) {
                assert(!submitted);
                assert(cudaLaunchHostFunc(gate_stream, hold_tail, nullptr) == cudaSuccess);
                assert(cudaEventRecord((cudaEvent_t) fence->context, gate_stream) == cudaSuccess);
                original.event_wait(backend, fence);
                submitted = true;
                break;
            }
        }
    }
    return status;
}

static void get_tensor(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (armed && strcmp(tensor->name, "result_output") == 0) {
        output_data = (float *) data;
        output_count = size / sizeof(float);
        std::fill(output_data, output_data + output_count, std::numeric_limits<float>::quiet_NaN());
        copied = true;
    }
    original.get_tensor_async(backend, tensor, data, offset, size);
}

static void synchronize(ggml_backend_t backend) {
    ++sync_count;
    if (armed && submitted && !saw_wait) {
        saw_wait = true;
        copy_before_wait = copied;
        pending_at_wait = cudaEventQuery((cudaEvent_t) fence->context) == cudaErrorNotReady;
        release_gate.store(true, std::memory_order_release);
    }
    original.synchronize(backend);
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    assert(common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER));
    // Match the one-slot, non-speculative server's output reservation.
    params.n_outputs_max = 1;
    auto init = common_init_from_params(params);
    auto * ctx = init->context();
    assert(ctx);
    auto sched = ctx->get_sched();
    ggml_backend_t gpu = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        auto backend = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            assert(!gpu);
            gpu = backend;
        }
    }
    assert(gpu);
    original = gpu->iface;
    profile = (ggml_backend_dispatch_profile_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)), "ggml_backend_dispatch_profile");
    assert(profile);
    assert(original.graph_compute && original.get_tensor_async && original.event_wait && original.synchronize);
    fence = ggml_backend_event_new(ggml_backend_get_device(gpu));
    assert(fence);
    assert(cudaStreamCreateWithFlags(&gate_stream, cudaStreamNonBlocking) == cudaSuccess);
    gpu->iface.graph_compute = graph_compute;
    gpu->iface.get_tensor_async = get_tensor;
    gpu->iface.synchronize = synchronize;

    llama_batch batch = llama_batch_init(2, 0, 1);
    common_batch_add(batch, 42, 0, {0}, false);
    common_batch_add(batch, 42, 1, {0}, true);
    assert(llama_decode(ctx, batch) == 0);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(init->model()));
    const bool expected = getenv("WACKMALL_TEST_OUTPUTS_BEFORE_SYNC") != nullptr;
    for (int step = -2; step < 34; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        assert(logits);
        llama_token next = 0;
        for (int i = 0; i < n_vocab; ++i) {
            assert(std::isfinite(logits[i]));
            if (logits[i] > logits[next]) {
                next = i;
            }
        }
        common_batch_clear(batch);
        common_batch_add(batch, next, step + 4, {0}, true);
        armed = true;
        submitted = copied = saw_wait = pending_at_wait = copy_before_wait = false;
        release_gate.store(false, std::memory_order_release);
        output_data = nullptr;
        // Fail safely even if a regression loses the completion hook.
        std::atomic<bool> finished {false};
        std::atomic<bool> timed_out {false};
        std::thread watchdog([&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!finished.load() && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (!finished.load()) {
                timed_out.store(true);
                release_gate.store(true, std::memory_order_release);
            }
        });
        const int result = step < 0 ? llama_decode(ctx, batch) : llama_decode_generated(ctx, batch);
        finished.store(true);
        watchdog.join();
        armed = false;
        assert(!timed_out.load() && result == 0 && submitted && copied && saw_wait && pending_at_wait);
        const bool reordered = expected && step >= 0;
        if (copy_before_wait != reordered) {
            fprintf(stderr, "FAIL: step %d outputs-before-tier-wait=%d, expected=%d; CUDA tail was pending\n", step, copy_before_wait, reordered);
            return 1;
        }
        assert(cudaEventQuery((cudaEvent_t) fence->context) == cudaSuccess);
        if (reordered) {
            // Check host data before any getter can provide a second wait.
            assert(output_data && output_count == (size_t) n_vocab);
            for (size_t i = 0; i < output_count; ++i) {
                assert(std::isfinite(output_data[i]));
            }
        }
        printf("PASS: step %d generated=%d token=%d copy-before-wait=%d pending-tail-drained=1\n", step + 1, step >= 0, next, copy_before_wait);
    }
    common_batch_clear(batch);
    common_batch_add(batch, 42, 38, {0}, false);
    const int waits_before = sync_count;
    assert(llama_decode_generated(ctx, batch) == 0);
    assert(sync_count > waits_before);
    common_batch_add(batch, 42, 39, {0}, true);
    assert(llama_decode_generated(ctx, batch) == -1);
    common_batch_clear(batch);
    common_batch_add(batch, 42, 39, {0}, true);
    assert(llama_decode_generated(ctx, batch) == 0);
    assert(llama_get_logits_ith(ctx, -1));
    puts("PASS: no-output completion and rejected multi-token attribution followed by valid decode");
    if (getenv("WACKMALL_TEST_ABORT")) {
        common_batch_clear(batch);
        common_batch_add(batch, 42, 40, {0}, true);
        ctx->set_abort_callback([](void *) { return true; }, nullptr);
        assert(llama_decode_generated(ctx, batch) == 2);
        ctx->set_abort_callback(nullptr, nullptr);
        // Recurrent state cannot promise partial rollback; explicitly drain and start a fresh sequence.
        llama_synchronize(ctx);
        llama_memory_clear(llama_get_memory(ctx), true);
        common_batch_clear(batch);
        common_batch_add(batch, 42, 0, {0}, true);
        assert(llama_decode(ctx, batch) == 0);
        const float * logits = llama_get_logits_ith(ctx, -1);
        assert(logits);
        llama_token next = 0;
        for (int i = 0; i < n_vocab; ++i) {
            assert(std::isfinite(logits[i]));
            if (logits[i] > logits[next]) next = i;
        }
        common_batch_clear(batch);
        common_batch_add(batch, next, 1, {0}, true);
        assert(llama_decode_generated(ctx, batch) == 0);
        assert(llama_get_logits_ith(ctx, -1));
        puts("PASS: real CPU graph abort, explicit completion/reset, fresh prompt and generated-decode recovery");
    }
    if (getenv("GGML_CUDA_DISABLE_GRAPHS")) {
        assert(reuses == 0 && launches == 0 && direct > 0);
    } else {
        assert(reuses > 0 && launches > 0);
    }
    printf("PASS: dispatch reuses=%d launches=%d direct=%d\n", reuses, launches, direct);
    gpu->iface = original;
    ggml_backend_event_free(fence);
    assert(cudaStreamDestroy(gate_stream) == cudaSuccess);
    llama_batch_free(batch);
    return 0;
}
