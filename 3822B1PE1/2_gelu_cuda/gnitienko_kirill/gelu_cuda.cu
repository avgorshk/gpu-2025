#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    constexpr float coef = 0.044715f;
    constexpr float scale = 0.7978845608f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float x2 = x * x;
        float inner = scale * (x + coef * x * x2);
        
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int size = static_cast<int>(input.size());
    if (size == 0) return {};
    
    float *data_input = nullptr, *data_output = nullptr;
    std::vector<float> result(size);
    
    cudaMalloc(&data_input, size * sizeof(float));
    cudaMalloc(&data_output, size * sizeof(float));
    cudaMemcpy(data_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    
    gelu_kernel<<<gridSize, blockSize>>>(data_input, data_output, size);
    
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(kernelError) << std::endl;
        cudaFree(data_input);
        cudaFree(data_output);
        return result;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(result.data(), data_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_input);
    cudaFree(data_output);
    
    return result;
}