#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>

constexpr float kGeluCoeff = 0.044715f;
constexpr float kSqrt2OverPi = 0.7978845608028654f;

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float x_cubed = x * x * x;
        float inner = kSqrt2OverPi * (x + kGeluCoeff * x_cubed);
        float tanh_val = tanhf(inner);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = static_cast<int>(input.size());
    std::vector<float> output(size);
    
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    
    cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}