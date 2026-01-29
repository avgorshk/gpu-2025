#include "gelu_cuda.h"
#include <cuda_runtime.h>

__constant__ float kSqrtTwoOverPi = 0.7978845608028654f;
__constant__ float kGeluCoeff = 0.044715f;

__global__ void GeluKernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        
        float inner = kSqrtTwoOverPi * (x + kGeluCoeff * x3);
        
        float exp_val = __expf(-2.0f * inner);
        float tanh_val = 1.0f - 2.0f / (1.0f + exp_val);
        
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = static_cast<int>(input.size());
    if (size == 0) {
        return std::vector<float>();
    }
    
    std::vector<float> output(size);
    
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpyAsync(d_input, input.data(), size * sizeof(float), 
                   cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    GeluKernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    
    cudaMemcpyAsync(output.data(), d_output, size * sizeof(float),
                   cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}