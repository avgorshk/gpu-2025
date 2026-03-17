#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void GeluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = static_cast<int>(input.size());
    std::vector<float> output(size);

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    const int kBlockSize = 256;
    int numBlocks = (size + kBlockSize - 1) / kBlockSize;
    GeluKernel<<<numBlocks, kBlockSize>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
