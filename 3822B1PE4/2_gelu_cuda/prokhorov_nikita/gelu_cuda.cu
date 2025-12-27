#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel_exact(const float* input, float* output, int n) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float x = input[idx];
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gelu_kernel_approx(const float* input, float* output, int n) {
    const float scale = 1.4142135623730951f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float x = input[idx];
    float cdf = 0.5f * (1.0f + erff(x / scale));
    output[idx] = x * cdf;
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    if (n == 0) return {};
    
    std::vector<float> result(n);
    size_t bytes = n * sizeof(float);
    
    float *d_input = nullptr, *d_output = nullptr;
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    
    gelu_kernel_exact<<<gridSize, blockSize>>>(d_input, d_output, n);
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        return {};
    }
    
    cudaMemcpy(result.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}