#include "gelu_cuda.h"
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = input[i];
    float x3 = x * x * x;
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float GELU_COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    output[i] = 0.5f * x * (1.0f + tanhf(inner));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    if (n == 0) return {};

    float* d_input = nullptr, * d_output = nullptr;
    size_t bytes = n * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int grid_size = (static_cast<int>(n) + block_size - 1) / block_size;
    gelu_kernel << <grid_size, block_size >> > (d_input, d_output, n);

    cudaDeviceSynchronize();

    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}