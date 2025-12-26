#include "gelu_cuda.h"
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void GeluKernel(const float *input, float *output, size_t size) {
    constexpr float SQRT_2_PI = 0.7978845608f; // sqrt(2 / pi)
    constexpr float COEF = 0.044715f;

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }

    float x = input[index];
    float arg = SQRT_2_PI * (x + COEF * x * x * x);
    output[index] = 0.5f * x * (1.0f + tanhf(arg));
}

std::vector<float> GeluCUDA(const std::vector<float> &input) {
    if (input.empty()) {
        return {};
    }

    size_t size = input.size();
    size_t data_size = size * sizeof(float);
    size_t block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;

    float *d_input = nullptr;
    cudaMalloc(&d_input, data_size);
    float *d_output = nullptr;
    cudaMalloc(&d_output, data_size);

    cudaMemcpy(d_input, input.data(), data_size, cudaMemcpyHostToDevice);
    GeluKernel<<<num_blocks, block_size>>>(d_input, d_output, size);
    std::vector<float> output(size);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_output, data_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}
