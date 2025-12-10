#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float c = 0.044715f;
    const float a = 0.79788456f;  // (2 / pi) ^ (1 / 2)
    if (idx < n) {
        float x = input[idx];
        float tanh_arg = a * x * (1.0f + c * x * x);
        float tanh = (exp(tanh_arg) - (1 / exp(tanh_arg))) / (exp(tanh_arg) + (1 / exp(tanh_arg)));
        output[idx] = 0.5f * x * (1.0f + tanh);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    if (n == 0) {
        return {};
    }

    size_t bytes = n * sizeof(float);
    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    gelu_kernel<<<blocks, threads_per_block>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return output;
}
