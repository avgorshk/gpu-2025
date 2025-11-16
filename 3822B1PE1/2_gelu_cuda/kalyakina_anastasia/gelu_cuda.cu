#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <iostream>

const float GELU_SCALE_FACTOR = 0.5f;
const float GELU_TANH_SCALE = 0.7978845608028654f;
const float GELU_CUBIC_COEFFICIENT = 0.044715f;

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float inner_value = GELU_TANH_SCALE * x * (1.0f + GELU_CUBIC_COEFFICIENT * x * x);
        float exp_value = __expf(2.0f * inner_value);
        float tanh_value = __fdividef(exp_value - 1.0f, exp_value + 1.0f);

        output[i] = GELU_SCALE_FACTOR * x * (1.0f + tanh_value);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> output(n);

    if (n == 0) return output;

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    gelu_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);

    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}