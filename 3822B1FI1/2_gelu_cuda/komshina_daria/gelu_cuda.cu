#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];
    const float c = 0.044715f;
    const float sqrt_2_over_pi = 0.7978845608f;

    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + c * x3);
    float exp_val = expf(-2.0f * inner);
    float tanh_approx = (1.0f - exp_val) / (1.0f + exp_val);
    output[idx] = 0.5f * x * (1.0f + tanh_approx);
}
std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    if (n == 0) return {};

    float* h_input = nullptr;
    float* h_output = nullptr;
    cudaMallocHost(&h_input, n * sizeof(float));
    cudaMallocHost(&h_output, n * sizeof(float));
    std::memcpy(h_input, input.data(), n * sizeof(float));

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(d_input, d_output, n);


    cudaMemcpyAsync(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    std::vector<float> result(n);
    std::memcpy(result.data(), h_output, n * sizeof(float));

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaStreamDestroy(stream);

    return result;
}
