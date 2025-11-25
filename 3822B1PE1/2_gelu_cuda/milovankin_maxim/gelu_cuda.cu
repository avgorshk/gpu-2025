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

__device__ float gelu_exp(float x) {
    const float c = 0.044715f;
    const float sqrt_2_over_pi = sqrtf(2.0f / 3.14159265358979323846f);
    float x3 = x * x * x;
    float z = sqrt_2_over_pi * (x + c * x3);
    // GELU = x * sigmoid(2*z) = x / (1 + exp(-2*z))
    float s = 1.0f / (1.0f + expf(-2.0f * z));
    return x * s;
}

__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gelu_exp(input[idx]);
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
