#include "gelu_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* input, float* output, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        const float half = 0.5f;
        
        float x = input[idx];
        float x_cubed = x * x * x;
        
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float exp_val = expf(-2.0f * inner);
        float tanh_val = (1.0f - exp_val) / (1.0f + exp_val);
        
        output[idx] = half * x * (1.0f + tanh_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> result(size);
    
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;
    
    gelu_kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);
    
    cudaMemcpy(result.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}