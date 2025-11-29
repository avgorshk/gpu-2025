#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void geluKernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const float sqrt_2_pi = 0.7978845608f;  // sqrt(2/π)
    const float coeff = 0.044715f;
    const float half = 0.5f;
    const float one = 1.0f;

    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = sqrt_2_pi * (x + coeff * x3);

        output[idx] = half * x * (one + std::tanh(inner));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> result(size);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    geluKernel<<<numBlocks, blockSize>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    cudaMemcpy(result.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}