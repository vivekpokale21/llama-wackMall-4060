#include "common.cuh"
#include "ggml-alloc.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <vector>

struct guarded_mmq_pool : ggml_cuda_pool {
    struct mapping { CUdeviceptr base; size_t mapped, reserved; };
    std::map<void *, mapping> allocations;

    void * alloc(size_t size, size_t * actual) override {
        size = GGML_PAD(size, 128);
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;
        size_t granularity;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        const size_t mapped = GGML_PAD(size, granularity);
        const size_t reserved = mapped + granularity;
        CUdeviceptr base;
        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemAddressReserve(&base, reserved, granularity, 0, 0));
        CU_CHECK(cuMemCreate(&handle, mapped, &prop, 0));
        CU_CHECK(cuMemMap(base, mapped, 0, handle, 0));
        CU_CHECK(cuMemRelease(handle));
        CUmemAccessDesc access = {};
        access.location = prop.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(cuMemSetAccess(base, mapped, &access, 1));
        void * ptr = reinterpret_cast<void *>(base + mapped - size);
        allocations[ptr] = {base, mapped, reserved};
        *actual = size;
        return ptr;
    }

    void free(void * ptr, size_t) override {
        const auto m = allocations.at(ptr);
        CU_CHECK(cuMemUnmap(m.base, m.mapped));
        CU_CHECK(cuMemAddressFree(m.base, m.reserved));
        allocations.erase(ptr);
    }
};

int main(int argc, char ** argv) {
    assert(argc == 3);
    const bool hot = strcmp(argv[1], "hot") == 0;
    const bool down = strcmp(argv[1], "down") == 0;
    assert(hot || down || strcmp(argv[1], "stock") == 0);
    const int tokens = std::atoi(argv[2]);
    assert(tokens > 0 && tokens <= 2048);
    constexpr int width = 2048, rows = 512, used = 8;
    const int experts = hot ? 19 : 256;
    const int channels = down ? used : 1;
    auto backend = ggml_backend_cuda_init(0);
    assert(backend);
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    cuda_ctx->pools[0][0] = std::make_unique<guarded_mmq_pool>();
    auto ctx = ggml_init({1024*1024, nullptr, true});
    auto weights = ggml_new_tensor_3d(ctx, GGML_TYPE_Q5_K, width, rows, experts);
    auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, channels, tokens);
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
    ggml_set_name(weights, hot ? "experts.hot" : "stock_experts");
    auto out = ggml_mul_mat_id(ctx, weights, x, ids);
    auto graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);

    std::vector<unsigned char> quantized(ggml_nbytes(weights));
    std::vector<float> row(width), dequantized(width);
    std::vector<double> weight_sums(experts);
    const size_t row_bytes = ggml_row_size(weights->type, width);
    for (int expert = 0; expert < experts; ++expert) {
        std::fill(row.begin(), row.end(), hot && expert == 18 ? 0.0f : (expert % 7 + 1)*0.25f);
        const size_t offset = expert * rows * row_bytes;
        const size_t written = ggml_quantize_chunk(weights->type, row.data(), quantized.data() + offset, 0, 1, width, nullptr);
        assert(written == row_bytes);
        ggml_get_type_traits(weights->type)->to_float(quantized.data() + offset, dequantized.data(), width);
        for (float value : dequantized) weight_sums[expert] += value;
        for (int r = 1; r < rows; ++r) {
            memcpy(quantized.data() + offset + r*row_bytes, quantized.data() + offset, row_bytes);
        }
    }
    std::vector<float> values(ggml_nelements(x));
    std::vector<int32_t> routes(ggml_nelements(ids));
    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < channels; ++c) {
            const float value = (t % 7 + 1)*(c % 3 + 1)*0.125f;
            std::fill_n(values.begin() + (t*channels+c)*width, width, value);
        }
        for (int k = 0; k < used; ++k) {
            routes[t*used+k] = hot ? (t%2 == 0 && k < 7 ? 18 : k) : (t*13+k*31)%256;
        }
    }
    std::vector<float> output(ggml_nelements(out), std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(x, values.data(), 0, values.size()*sizeof(float));
    ggml_backend_tensor_set(ids, routes.data(), 0, routes.size()*sizeof(int32_t));
    ggml_backend_tensor_set(out, output.data(), 0, output.size()*sizeof(float));
    const auto status = ggml_backend_graph_compute(backend, graph);
    assert(status == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(out, output.data(), 0, output.size()*sizeof(float));
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < used; ++k) {
            const double expected = weight_sums[routes[t*used+k]] * values[(t*channels + k%channels)*width];
            for (int r = 0; r < rows; ++r) {
                const float value = output[(t*used+k)*rows+r];
                assert(std::isfinite(value));
                assert(std::abs(value-expected) <= std::max(1e-3, std::abs(expected)*1e-3));
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    ggml_quantize_free();
    printf("PASS: guarded %s/%d, nonzero CPU-reference sums and zero sentinel\n", argv[1], tokens);
}
