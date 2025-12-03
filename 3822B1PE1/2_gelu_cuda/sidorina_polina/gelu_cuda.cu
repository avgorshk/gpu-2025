#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <iostream>

__global__ void GeluKernel(const float* input, float* output, int num_elements)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements)
    {
        const float x = input[idx];
        const float x_cubed = x * x * x;
        const float inner = 0.7978845608028654f * (x + 0.044715f * x_cubed);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) 
{
    const size_t num_elements = input.size();
    const size_t data_size = num_elements * sizeof(float);
    
    constexpr int THREADS_PER_BLOCK = 256;
    const int blocks_per_grid = (static_cast<int>(num_elements) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    std::vector<float> result(num_elements);
    
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size);
    
    cudaMemcpy(d_input, input.data(), data_size, cudaMemcpyHostToDevice);
    
    GeluKernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_input, d_output, static_cast<int>(num_elements));
    
    cudaMemcpy(result.data(), d_output, data_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}