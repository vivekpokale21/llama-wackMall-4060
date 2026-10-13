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

// Test-only real CUDA readers held outstanding until the tier's completion wait.
class wackmall_async_reader {
    struct snapshot {
        std::vector<char> expected;
        void * actual = nullptr;
    };
    ggml_backend_t backend;
    void (*original_sync)(ggml_backend_t);
    ggml_backend_event_t fence;
    cudaStream_t stream;
    std::atomic<bool> release {false}, finished {false}, timed_out {false};
    std::thread watchdog;
    std::vector<snapshot> copies;
    static wackmall_async_reader * active;
    bool saw_pending = false;

    static void CUDART_CB hold(void * data) {
        auto * self = (wackmall_async_reader *) data;
        while (!self->release.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
    static void synchronize(ggml_backend_t backend) {
        auto & self = *active;
        assert(backend == self.backend);
        self.saw_pending = cudaEventQuery((cudaEvent_t) self.fence->context) == cudaErrorNotReady;
        self.release.store(true, std::memory_order_release);
        self.original_sync(backend);
        // Check the outstanding reader before the hook can replace weights or LUTs.
        for (const auto & copy : self.copies) {
            assert(memcmp(copy.expected.data(), copy.actual, copy.expected.size()) == 0);
        }
    }

public:
    wackmall_async_reader(ggml_backend_t backend, const std::vector<ggml_tensor *> & tensors) : backend(backend) {
        assert(!active && !tensors.empty());
        fence = ggml_backend_event_new(ggml_backend_get_device(backend));
        assert(fence);
        assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
        for (auto * tensor : tensors) {
            snapshot copy;
            copy.expected.resize(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, copy.expected.data(), 0, copy.expected.size());
            assert(cudaMallocHost(&copy.actual, copy.expected.size()) == cudaSuccess);
            copies.push_back(std::move(copy));
        }
        assert(cudaLaunchHostFunc(stream, hold, this) == cudaSuccess);
        for (size_t i = 0; i < tensors.size(); ++i) {
            assert(cudaMemcpyAsync(copies[i].actual, tensors[i]->data, copies[i].expected.size(), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
        }
        assert(cudaEventRecord((cudaEvent_t) fence->context, stream) == cudaSuccess);
        ggml_backend_event_wait(backend, fence);
        original_sync = backend->iface.synchronize;
        active = this;
        backend->iface.synchronize = synchronize;
        watchdog = std::thread([this] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!finished.load() && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (!finished.load()) {
                timed_out.store(true);
                release.store(true, std::memory_order_release);
            }
        });
    }
    ~wackmall_async_reader() {
        finished.store(true);
        watchdog.join();
        assert(saw_pending && !timed_out.load());
        backend->iface.synchronize = original_sync;
        active = nullptr;
        ggml_backend_event_free(fence);
        assert(cudaStreamDestroy(stream) == cudaSuccess);
        for (const auto & copy : copies) {
            assert(cudaFreeHost(copy.actual) == cudaSuccess);
        }
    }
};

wackmall_async_reader * wackmall_async_reader::active = nullptr;
