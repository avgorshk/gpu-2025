#include "gelu_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* input, float* output, int N, float sqrt_pi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float input_value = input[idx];
        float arg_tanh = sqrt_pi * (input_value + 0.044715f * input_value * input_value * input_value);
        float exp_term = exp(2.0f * arg_tanh);
        float tanh = (exp_term - 1.0f) / (exp_term + 1.0f);
        output[idx] = 0.5f * input_value * (1.0f + tanh);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t N = input.size();
    std::vector<float> output(N);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);

    const float sqrt_pi = sqrtf(2.0f / acosf(-1.0f));

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    gelu_kernel<<<blocks, threads_per_block, 0, stream>>>(d_input, d_output, static_cast<int>(N), sqrt_pi);

    cudaMemcpyAsync(output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}