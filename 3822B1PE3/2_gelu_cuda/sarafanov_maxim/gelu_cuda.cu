#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        // Fast GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        const float k = 0.70710678f; // 1/sqrt(2)
        out[i] = 0.5f * x * (1.0f + erff(x * k));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    if (n == 0) return {};

    float *d_in = nullptr;
    float *d_out = nullptr;

    cudaMalloc(&d_in, sizeof(float) * n);
    cudaMalloc(&d_out, sizeof(float) * n);

    cudaMemcpyAsync(d_in, input.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;

    gelu_kernel<<<grid, block>>>(d_in, d_out, n);

    std::vector<float> result(n);
    cudaMemcpyAsync(result.data(), d_out, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);

    return result;
}
