#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_SAFE(x) do {                                          \
    cudaError_t e = (x);                                           \
    if (e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA failure at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(e));        \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

__device__ float approx_gelu(float v) {
    const float poly_k = 0.044715f;
    const float inv_sqrt_pi2 = sqrtf(2.0f / 3.14159265358979323846f);

    float v3 = v * v * v;
    float arg = inv_sqrt_pi2 * (v + poly_k * v3);

    float sig = 1.0f / (1.0f + expf(-2.0f * arg));
    return v * sig;
}

__global__ void kernel_gelu(const float* src, float* dst, size_t count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        dst[tid] = approx_gelu(src[tid]);
    }
}

std::vector<float> computeGeluGPU(const std::vector<float>& data) {
    size_t count = data.size();
    if (count == 0) return {};

    size_t mem = count * sizeof(float);
    float* d_src = nullptr;
    float* d_dst = nullptr;

    CUDA_SAFE(cudaMalloc(&d_src, mem));
    CUDA_SAFE(cudaMalloc(&d_dst, mem));

    CUDA_SAFE(cudaMemcpy(d_src, data.data(), mem, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (count + blockSize - 1) / blockSize;

    kernel_gelu<<<gridSize, blockSize>>>(d_src, d_dst, count);
    CUDA_SAFE(cudaGetLastError());
    CUDA_SAFE(cudaDeviceSynchronize());

    std::vector<float> result(count);
    CUDA_SAFE(cudaMemcpy(result.data(), d_dst, mem, cudaMemcpyDeviceToHost));

    CUDA_SAFE(cudaFree(d_src));
    CUDA_SAFE(cudaFree(d_dst));

    return result;
}
