#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gelu_cuda.h"

__global__ void GeluKernel(const float* input, float* output, size_t size)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size)
        return;

    constexpr float coefficient1 = 1.595769122f;
    constexpr float coefficient2 = 0.071354816f;

    float value = input[index];
    output[index] = value * (1.0f - 1.0f / (1.0f + __expf(value * (coefficient1 + value * value * coefficient2))));
}

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    if (input.empty())
        return {};

    size_t data_size = input.size();
    std::vector<float> output(data_size);

    size_t bytes_size = data_size * sizeof(float);
    size_t threads_per_block = 256;
    size_t blocks_count = (data_size + threads_per_block - 1) / threads_per_block;

    float* device_input = nullptr;
    cudaMalloc(&device_input, bytes_size);

    float* device_output = nullptr;
    cudaMalloc(&device_output, bytes_size);

    cudaMemcpy(device_input, input.data(), bytes_size, cudaMemcpyHostToDevice);

    GeluKernel<<<blocks_count, threads_per_block>>>(device_input, device_output, data_size);

    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), device_output, bytes_size, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    return output;
}