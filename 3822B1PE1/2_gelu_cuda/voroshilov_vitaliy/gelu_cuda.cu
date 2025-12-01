#include "gelu_cuda.h"

#include <cmath>

#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float x = input[idx];
    float x3 = x * x * x;
    const float sqrt2pi = sqrtf(2.0 / M_PI);
    output[idx] = 0.5 * x * (1 + tanhf(sqrt2pi * (x + 0.044715 * x3)));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    size_t n = input.size();
    std::vector<float> output(n);

    if (n == 0) {
        return output;
    }

    float *d_input = nullptr;
    float *d_output = nullptr;
    size_t n_bytes = n * sizeof(float);

    cudaMalloc((void**)&d_input, n_bytes);
    cudaMalloc((void**)&d_output, n_bytes);

    cudaMemcpy(d_input, input.data(), n_bytes, cudaMemcpyHostToDevice);

    int threads = prop.maxThreadsPerBlock;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gelu_kernel<<<blocks, threads>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
