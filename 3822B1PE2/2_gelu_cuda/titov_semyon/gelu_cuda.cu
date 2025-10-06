#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__global__ void GeluKernel(const float* input, float* output, int n) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coef * x_cubed);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> output(n);

    if (n == 0) return output;

    float* d_input, * d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    GeluKernel << <numBlocks, blockSize >> > (d_input, d_output, n);

    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}