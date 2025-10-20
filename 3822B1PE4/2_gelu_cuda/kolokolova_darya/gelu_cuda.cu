#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float pi = 3.14159265358979323846f;
        float sqrt_2_over_pi = sqrtf(2.0f / pi);
        float coefficient = 0.044715f;
        
        // Вычисление GELU
        float inner = x + coefficient * x * x * x;
        float tanh_value = tanhf(sqrt_2_over_pi * inner);
        output[idx] = 0.5f * x * (1.0f + tanh_value);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();
    std::vector<float> result(size);
    
    if (size == 0) return result;
    
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(d_input, d_output, size);
    
    cudaMemcpy(result.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}