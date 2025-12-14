#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x2 = x * x;
        float x3 = x2 * x;

        constexpr float kAlpha = 0.044715f;
        constexpr float kSqrt2toPi = 0.7978845608028654f;

        float t = kSqrt2toPi * (x + kAlpha * x3);
        float exp_term = expf(-2.0f * t);
        float tanh_approx = 2.0f / (1.0f + exp_term) - 1.0f;

        output[idx] = 0.5f * x * (1.0f + tanh_approx);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    float *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    gelu_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
