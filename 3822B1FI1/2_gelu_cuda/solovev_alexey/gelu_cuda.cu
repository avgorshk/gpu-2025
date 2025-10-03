#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__device__ float gelu_func(float x) {
    const float k = sqrtf(2.0f / 3.14159265f);
    float inner = k * (x + 0.044715f * x * x * x);
    float e = expf(-2.0f * inner);
    float t = (1.0f - e) / (1.0f + e);
    return 0.5f * x * (1.0f + t);
}

__global__ void gelu_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gelu_func(in[i]);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    if (n == 0) return {};
    size_t bytes = n * sizeof(float);

    float* d_in, * d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel << <blocks, threads >> > (d_in, d_out, n);

    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return output;
}