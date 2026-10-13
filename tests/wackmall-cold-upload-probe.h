#pragma once

#include "ggml-backend-impl.h"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#undef NDEBUG
#include <cassert>

class wackmall_cold_upload_probe {
    ggml_backend_t gpu;
    ggml_tensor * source;
    ggml_backend_i original;
    ggml_backend_event_t fence;
    cudaStream_t stream;
    std::atomic<bool> release {false}, finished {false}, timed_out {false};
    std::thread watchdog;
    std::vector<char> expected;
    void * actual = nullptr;
    bool submitted = false, drained = false, aborted = false;
    bool abort_on_submit;
    static wackmall_cold_upload_probe * active;
    static void CUDART_CB hold(void * data) {
        auto & self = *(wackmall_cold_upload_probe *) data;
        while (!self.release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    static void set(ggml_backend_t gpu, ggml_tensor * dst, const void * data, size_t offset, size_t size) {
        auto & self = *active;
        if (data == self.source->data && !self.submitted) {
            assert(self.source->flags & GGML_TENSOR_FLAG_OUTPUT);
            assert(offset == 0 && size == ggml_nbytes(self.source));
            memcpy(self.expected.data(), data, size);
            assert(cudaLaunchHostFunc(self.stream, hold, &self) == cudaSuccess);
            assert(cudaEventRecord((cudaEvent_t) self.fence->context, self.stream) == cudaSuccess);
            self.original.event_wait(gpu, self.fence);
            self.submitted = true;
            self.original.set_tensor_async(gpu, dst, data, offset, size);
            self.original.get_tensor_async(gpu, dst, self.actual, 0, size);
        } else self.original.set_tensor_async(gpu, dst, data, offset, size);
    }
    static void synchronize(ggml_backend_t gpu) {
        auto & self = *active;
        if (self.submitted && !self.drained) {
            assert(cudaEventQuery((cudaEvent_t) self.fence->context) == cudaErrorNotReady);
            assert(memcmp(self.expected.data(), self.source->data, self.expected.size()) == 0);
            self.release.store(true);
            self.drained = true;
        }
        self.original.synchronize(gpu);
        if (self.drained) assert(memcmp(self.expected.data(), self.actual, self.expected.size()) == 0);
    }
    static ggml_status graph_compute(ggml_backend_t gpu, ggml_cgraph * graph) {
        auto & self = *active;
        if (self.abort_on_submit && self.submitted && !self.aborted) {
            self.aborted = true;
            return GGML_STATUS_ABORTED;
        }
        return self.original.graph_compute(gpu, graph);
    }
public:
    wackmall_cold_upload_probe(ggml_backend_t gpu, ggml_tensor * source, bool abort_on_submit) :
            gpu(gpu), source(source), original(gpu->iface), abort_on_submit(abort_on_submit) {
        assert(!active && original.set_tensor_async && original.get_tensor_async && original.synchronize && original.event_wait);
        expected.resize(ggml_nbytes(source));
        assert(cudaMallocHost(&actual, expected.size()) == cudaSuccess);
        assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
        fence = ggml_backend_event_new(ggml_backend_get_device(gpu));
        assert(fence);
        active = this;
        gpu->iface.set_tensor_async = set;
        gpu->iface.synchronize = synchronize;
        gpu->iface.graph_compute = graph_compute;
        watchdog = std::thread([this] {
            const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!finished.load() && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!finished.load()) { timed_out.store(true); release.store(true); }
        });
    }
    ~wackmall_cold_upload_probe() {
        finished.store(true);
        watchdog.join();
        assert(submitted && drained && !timed_out.load());
        gpu->iface = original;
        active = nullptr;
        ggml_backend_event_free(fence);
        assert(cudaStreamDestroy(stream) == cudaSuccess);
        assert(cudaFreeHost(actual) == cudaSuccess);
    }
    void pending() const {
        assert(submitted && !drained && !timed_out.load());
        assert(cudaEventQuery((cudaEvent_t) fence->context) == cudaErrorNotReady);
    }
};

wackmall_cold_upload_probe * wackmall_cold_upload_probe::active = nullptr;
