#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

__global__ void geluKernel(const float* __restrict__ input, 
                           float* __restrict__ output, 
                           size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoeff = 0.044715f;

    float x = input[idx];
    float xCubed = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * xCubed);
    float tanhVal = tanhf(inner);
    output[idx] = 0.5f * x * (1.0f + tanhVal);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);
    
    if (n == 0) {
        return output;
    }

    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;

    cudaError_t err = cudaMalloc(&deviceInput, n * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for input");
    }

    err = cudaMalloc(&deviceOutput, n * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(deviceInput);
        throw std::runtime_error("Failed to allocate device memory for output");
    }

    cudaMemcpy(deviceInput, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    geluKernel<<<gridSize, blockSize>>>(deviceInput, deviceOutput, n);

    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), deviceOutput, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return output;
}
