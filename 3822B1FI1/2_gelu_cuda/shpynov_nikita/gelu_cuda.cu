#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <string>
#include <sstream>

__constant__ float SQRT_2_OVER_PI = 0.797885f;
__constant__ float COEF = 0.044715f;

__global__ void gelu_kernel(const float *input, float *output, size_t size)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < size; i += stride)
    {
        const float x = input[i];
        const float x_cubed = x * x * x;
        const float inner = SQRT_2_OVER_PI * (x + COEF * x_cubed);
        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

cudaError_t launch_gelu_kernel(const float *d_input, float *d_output, size_t size,
                               cudaStream_t stream = 0)
{

    int block_size = 256;
    int min_grid_size;
    int grid_size;

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       gelu_kernel,
                                       0, 0);

    grid_size = (size + block_size - 1) / block_size;
    gelu_kernel<<<grid_size, block_size, 0, stream>>>(d_input, d_output, size);

    return cudaPeekAtLastError();
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    std::vector<float> output(input.size());
    float *d_input = nullptr;
    float *d_output = nullptr;

    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_output, input.size() * sizeof(float));

    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    int block_size = 512;
    int blocks_amount = (input.size() + block_size - 1) / block_size;
    gelu_kernel<<<blocks_amount, block_size>>>(d_input, d_output, input.size());

    cudaMemcpy(output.data(), d_output, input.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}