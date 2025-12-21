#include <cuda_runtime.h>
#include <cmath>
#include "gelu_cuda.h"

__global__ void gelu_kernel(const float* input, float* output, int n, float sqrt2OverPi, float coefficient) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float x3 = x * x * x;
        float arg = sqrt2OverPi * (x + coefficient * x3);
        float exp2arg = expf(2.0f * arg);
        float tanhValue = (exp2arg - 1.0f) / (exp2arg + 1.0f);
        output[i] = 0.5f * x * (1.0f + tanhValue);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> output(n);

    float sqrt2OverPi = std::sqrt(2.0f / acosf(-1.0f));
    float coefficient = 0.044715f;

    float *deviceInput, *deviceOutput;
    cudaMalloc(&deviceInput, n * sizeof(float));
    cudaMalloc(&deviceOutput, n * sizeof(float));

    cudaMemcpy(deviceInput, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, gelu_kernel, 0, 0);

    int numBlocks = (n + blockSize - 1) / blockSize;
    gelu_kernel<<<numBlocks, blockSize>>>(deviceInput, deviceOutput, n, sqrt2OverPi, coefficient);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), deviceOutput, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return output;
}