#include "gelu_cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

constexpr float SQRT_2_OVER_PI = 0.7978845608f;
constexpr float GELU_COEFF     = 0.044715f;

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x  = input[idx];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    std::vector<float> output(n);
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(output.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return output;
}