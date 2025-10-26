#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const float x = input[idx];
        const float x_cube = x * x * x;
        const float alpha = 0.044715f;
        const float beta = 0.7978845608028654f; // sqrt(2/π)
        
        const float inner = beta * (x + alpha * x_cube);
        const float exp_val = expf(-2.0f * inner);
        const float tanh_approx = 1.0f - 2.0f / (1.0f + exp_val);
        
        output[idx] = 0.5f * x * (1.0f + tanh_approx);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int size = input.size();
    std::vector<float> output(size);
    
    if (size == 0) {
        return output;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    
    gelu_kernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    
    cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}