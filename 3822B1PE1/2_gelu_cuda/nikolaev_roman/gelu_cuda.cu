#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <stdexcept>

constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
constexpr float GELU_CONST = 0.044715f;

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_CONST * x3);
        float tanh_val = tanhf(inner);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();
    std::vector<float> result(size);
    
    if (size == 0) return result;
    
    float *d_input, *d_output;
    cudaError_t err;
    
    err = cudaMalloc(&d_input, size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("CUDA malloc failed for input");
    
    err = cudaMalloc(&d_output, size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        throw std::runtime_error("CUDA malloc failed for output");
    }
    
    err = cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error("CUDA memcpy failed");
    }
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    gelu_kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error("CUDA kernel launch failed");
    }
    
    err = cudaMemcpy(result.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error("CUDA memcpy back failed");
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}