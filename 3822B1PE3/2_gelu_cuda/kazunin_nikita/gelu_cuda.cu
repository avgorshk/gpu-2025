#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <iostream>

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x  = input[idx];
    float x3 = x * x * x;
    const float k_sqrt2_over_pi = 0.7978845608f;
    const float k_cubic_coeff   = 0.044715f;
    float y  = k_sqrt2_over_pi * (x + k_cubic_coeff * x3);

    float t  = tanhf(y);
    output[idx] = 0.5f * x * (1.0f + t);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);

    if (n == 0) return output;

    float* d_input  = nullptr;
    float* d_output = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_input, n * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate device memory for input");

    err = cudaMalloc(&d_output, n * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate device memory for output");

    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize  = (n + blockSize - 1) / blockSize;
    gelu_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    cudaDeviceSynchronize();ч

    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
