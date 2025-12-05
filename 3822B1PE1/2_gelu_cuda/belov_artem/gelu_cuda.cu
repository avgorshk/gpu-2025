#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        const float x = input[idx];
        const float x_cubed = x * x * x;
        
        constexpr float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
        constexpr float coeff = 0.044715f;
        
        const float arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        const float exp_2arg = expf(2.0f * arg);
        const float tanh_arg = (exp_2arg - 1.0f) / (exp_2arg + 1.0f);
        
        output[idx] = 0.5f * x * (1.0f + tanh_arg);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    
    static float* d_input = nullptr;
    static float* d_output = nullptr;
    static size_t allocated_size = 0;
    
    if (allocated_size < n) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        allocated_size = n;
    }
    
    std::vector<float> output(n);
    
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    gelu_kernel<<<blocks, threads_per_block>>>(d_input, d_output, n);
    
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    return output;
}
