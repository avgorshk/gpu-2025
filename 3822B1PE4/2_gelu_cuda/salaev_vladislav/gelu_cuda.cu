#include "gelu_cuda.h"
#include <cuda_runtime.h>

__global__ void geluKernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        
        float exp2x = __expf(2.0f * inner);
        float tanh_val = (exp2x - 1.0f) / (exp2x + 1.0f);
        
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    geluKernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    
    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}