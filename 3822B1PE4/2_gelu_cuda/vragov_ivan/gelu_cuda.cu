#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// Constants for GELU computation
constexpr float GELU_SQRT_2_OVER_PI = 0.7978845608028654f;
constexpr float GELU_COEFF = 0.044715f;
constexpr float GELU_HALF = 0.5f;

// CUDA kernel for GELU computation
__global__ void gelu_kernel(const float* __restrict__ input, 
                            float* __restrict__ output, 
                            size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
        // Use exp instead of tanh for better performance
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        float exp_2x = expf(2.0f * inner);
        float tanh_val = (exp_2x - 1.0f) / (exp_2x + 1.0f);
        output[idx] = GELU_HALF * x * (1.0f + tanh_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    
    if (n == 0) {
        return std::vector<float>();
    }
    
    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    size_t bytes = n * sizeof(float);
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (static_cast<int>(n) + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    // Allocate host output and copy result back
    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

