#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
constexpr float GELU_COEFF = 0.044715f;
constexpr float HALF = 0.5f;

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float x_cubed = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
        float tanh_val = tanhf(inner);
        output[idx] = HALF * x * (1.0f + tanh_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = static_cast<int>(input.size());
    std::vector<float> output(size);
    
    if (size == 0) return output;
    
    float *d_input = nullptr, *d_output = nullptr;
    
    cudaError_t err = cudaMalloc(&d_input, size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc error (input): " << cudaGetErrorString(err) << std::endl;
        return output;
    }
    
    err = cudaMalloc(&d_output, size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc error (output): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return output;
    }
    
    err = cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Memcpy HtoD error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return output;
    }
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Device Synchronize error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Memcpy DtoH error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}
