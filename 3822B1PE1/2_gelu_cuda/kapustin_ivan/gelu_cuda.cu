#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = input[i];
    float x_cubed = x * x * x;

    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;

    float gelu_arg = sqrt_2_over_pi * (x + coeff * x_cubed);

    float exp_term = expf(2.0f * gelu_arg);
    float tanh_val = (exp_term - 1.0f) / (exp_term + 1.0f);

    output[i] = 0.5f * x * (1.0f + tanh_val);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static size_t allocated = 0;

    if (allocated < n) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);

        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        allocated = n;
    }

    std::vector<float> output(n);

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(d_in, d_out, n);

    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return output;
}