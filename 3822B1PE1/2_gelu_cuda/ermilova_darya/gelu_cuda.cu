#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

constexpr float SQRT_2_OVER_PI = 0.7978845608f; // sqrt(2/pi)
constexpr float GELU_COEFF = 0.044715f;

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    float gelu = 0.5f * x * (1.0f + tanhf(inner));

    output[idx] = gelu;
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    if (n == 0) return {};

    std::vector<float> output(n);
    float* d_input = nullptr, * d_output = nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);


    cudaError_t err;
    err = cudaMalloc(&d_input, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_input error: " << cudaGetErrorString(err) << std::endl;
        return {};
    }
    err = cudaMalloc(&d_output, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_output error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return {};
    }

    err = cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return {};
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel << <blocks, threads >> > (d_input, d_output, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return {};
    }

    cudaDeviceSynchronize();

    err = cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
