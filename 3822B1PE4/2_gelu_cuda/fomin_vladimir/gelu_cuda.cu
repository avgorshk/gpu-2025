#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__global__ void gelu_kernel(const float *__restrict__ input, float *__restrict__ output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    const float x = input[idx];
    const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
    const float coeff = 0.044715f;

    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    float tanh_val = __tanhf(inner);
    output[idx] = 0.5f * x * (1.0f + tanh_val);
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    size_t n = input.size();
    if (n == 0)
        return std::vector<float>();

    std::vector<float> output(n);

    float *d_input = nullptr, *d_output = nullptr;

    // Allocate device memory
    size_t bytes = n * sizeof(float);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    int grid_size = (static_cast<int>(n) + block_size - 1) / block_size;

    gelu_kernel<<<grid_size, block_size>>>(d_input, d_output, n);

    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}