#include <cuda_runtime.h>
#include <cmath>
#include "gelu_cuda.h"

__constant__ float sqrt_2_pi;
__constant__ float coeff = 0.044715f;

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        float arg = sqrt_2_pi * (x + coeff * x3);
        float exp2z = expf(2.0f * arg);
        float tanh_val = (exp2z - 1.0f) / (exp2z + 1.0f);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> output(n);

    float val1 = std::sqrt(2.0f / acosf(-1.0f));
    cudaMemcpyToSymbol(sqrt_2_pi, &val1, sizeof(float));

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, gelu_kernel, 0, 0);

    int numBlocks = (n + blockSize - 1) / blockSize;
    gelu_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}