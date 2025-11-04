#include "gelu_cuda.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

__global__ void kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

    auto gelu = [] __device__ (float x) __forceinline__ -> float {
        const float c = 0.044715f;
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float x3 = x * x * x;
        float z  = sqrt_2_over_pi * (x + c * x3);
        float s  = 1.0f / (1.0f + expf(-2.0f * z));
        return x * s;
    };

    for (size_t i = idx; i < n; i += stride) {
        out[i] = gelu(in[i]);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();

    size_t bytes = n * sizeof(float);
    float *d_in = nullptr;
    float *d_out = nullptr;

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    const int threads = 256;
    size_t blocks_needed = (n + threads - 1) / threads;
    size_t blocks = std::min<size_t>(blocks_needed, 65535);

    kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaGetLastError();
    cudaDeviceSynchronize();

    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return output;
}