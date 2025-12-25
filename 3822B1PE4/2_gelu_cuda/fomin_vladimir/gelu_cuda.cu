#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__global__ void gelu_kernel(const float *__restrict__ input, float *__restrict__ output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    const float x = input[idx];
    const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
    const float coeff = 0.044715f;

    float inner = x + coeff * x * x * x;
    float tanh_val = tanhf(sqrt_2_over_pi * inner);
    output[idx] = 0.5f * x * (1.0f + tanh_val);
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    size_t n = input.size();
    if (n == 0)
        return {};

    if (n > static_cast<size_t>(INT_MAX))
    {
        throw std::runtime_error("Input size exceeds maximum supported by CUDA kernel (INT_MAX).");
    }

    std::vector<float> output(n);
    float *d_input = nullptr, *d_output = nullptr;

    size_t bytes = n * sizeof(float);
    cudaError_t err;

    err = cudaMalloc(&d_input, bytes);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaMalloc failed for d_input: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_output, bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_input);
        throw std::runtime_error("cudaMalloc failed for d_output: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error("cudaMemcpy to device failed: " + std::string(cudaGetErrorString(err)));
    }

    const int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    gelu_kernel<<<grid_size, block_size>>>(d_input, d_output, static_cast<int>(n));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error("cudaMemcpy from device failed: " + std::string(cudaGetErrorString(err)));
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}