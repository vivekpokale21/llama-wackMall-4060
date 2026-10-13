#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)

struct payload { float x[32]; int ids[8]; };
struct gate { std::atomic<bool> release {false}; std::atomic<bool> timed_out {false}; };
static void CUDART_CB block_suffix(void * data) {
    auto * g = static_cast<gate *>(data);
    while (!g->release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

int main(int argc, char **) {
    const bool external = argc == 1;
    payload * input, * staged, * device;
    CHECK(cudaMallocHost(&input, sizeof(payload)));
    CHECK(cudaMallocHost(&staged, sizeof(payload)));
    CHECK(cudaMalloc(&device, sizeof(payload)));
    cudaStream_t stream;
    cudaEvent_t ready, done;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    gate g;
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    CHECK(cudaMemcpyAsync(device, input, sizeof(payload), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(staged->x, device->x, sizeof(input->x), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(staged->ids, device->ids, sizeof(input->ids), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaEventRecordWithFlags(ready, stream, external ? cudaEventRecordExternal : cudaEventRecordDefault));
    CHECK(cudaLaunchHostFunc(stream, block_suffix, &g));
    CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t instance;
    CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    bool valid = true;
    for (int generation = 1; generation <= 16; ++generation) {
        for (int i = 0; i < 32; ++i) { input->x[i] = generation + i * 0.25f; staged->x[i] = -1; }
        for (int i = 0; i < 8; ++i) { input->ids[i] = generation * 8 + i; staged->ids[i] = -1; }
        g.release = false;
        g.timed_out = false;
        std::thread watchdog([&] {
            const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!g.release && std::chrono::steady_clock::now() < end) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (!g.release.exchange(true)) g.timed_out = true;
        });
        CHECK(cudaGraphLaunch(instance, stream));
        CHECK(cudaEventRecord(done, stream));
        CHECK(cudaEventSynchronize(ready));
        for (int i = 0; i < 32; ++i) valid &= staged->x[i] == input->x[i];
        for (int i = 0; i < 8; ++i) valid &= staged->ids[i] == input->ids[i];
        valid &= cudaEventQuery(done) == cudaErrorNotReady;
        valid &= !g.timed_out.load();
        g.release = true;
        watchdog.join();
        CHECK(cudaStreamSynchronize(stream));
        if (!valid) break;
    }
    CHECK(cudaGraphExecDestroy(instance));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaEventDestroy(ready));
    CHECK(cudaEventDestroy(done));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(device)); CHECK(cudaFreeHost(input)); CHECK(cudaFreeHost(staged));
    if (!valid) { fputs("FAIL: current-generation prefix ready before suffix completion\n", stderr); return 1; }
    puts("PASS: 16 captured launches, current F32/I32 input, prefix ready while suffix blocked");
}
