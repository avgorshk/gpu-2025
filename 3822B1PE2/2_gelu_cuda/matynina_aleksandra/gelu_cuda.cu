#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cmath>
#include <iostream>

__constant__ float c_sqrt_2_over_pi = 0.7978845608028654f;
__constant__ float c_coeff = 0.044715f;
__constant__ float c_half = 0.5f;

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        float arg = c_sqrt_2_over_pi * (x + c_coeff * x3);
        float exp2arg = expf(2.0f * arg);
        float tanh_val = (exp2arg - 1.0f) / (exp2arg + 1.0f);
        output[idx] = c_half * x * (1.0f + tanh_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t size = input.size();
    
    if (size == 0) {
        return std::vector<float>();
    }
    
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    size_t bytes = size * sizeof(float);
    
    cudaError_t err = cudaMalloc(&d_input, bytes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for input: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float>();
    }
    
    err = cudaMalloc(&d_output, bytes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return std::vector<float>();
    }
    
    err = cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed (H2D): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return std::vector<float>();
    }
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (static_cast<int>(size) + threadsPerBlock - 1) / threadsPerBlock;
    
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, static_cast<int>(size));
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return std::vector<float>();
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return std::vector<float>();
    }
    
    std::vector<float> output(size);
    err = cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed (D2H): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return std::vector<float>();
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

