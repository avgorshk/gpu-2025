#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <stdexcept>

__device__ __forceinline__ float gelu_single(float x) {
    const float alpha = sqrtf(2.0f / 3.14159265f);
    const float c = 0.044715f;
    float x3 = x * x * x;
    float t = alpha * (x + c * x3);
    float tanh_t = tanhf(t);
    return 0.5f * x * (1.0f + tanh_t);
}

__global__ void gelu_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = gelu_single(in[idx]);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    if (n == 0) return {};

    size_t bytes = n * sizeof(float);
    float *d_in = nullptr, *d_out = nullptr;

    if (cudaMalloc(&d_in, bytes) != cudaSuccess ||
        cudaMalloc(&d_out, bytes) != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed");
    }

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    gelu_kernel<<<gridSize, blockSize>>>(d_in, d_out, n);

    cudaDeviceSynchronize();

    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return output;
}
