#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <vector>

__global__ void GeluKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x2 = x * x;
        float x3 = x2 * x;
        float u = 0.7978845608f * (x + 0.044715f * x3);
        float t = tanhf(u);
        output[idx] = 0.5f * x * (1.0f + t);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<float> output(n);
    if (n == 0) {
        return output;
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, static_cast<size_t>(n) * sizeof(float));
    cudaMalloc(&d_output, static_cast<size_t>(n) * sizeof(float));

    cudaMemcpy(d_input, input.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    GeluKernel<<<grid_size, block_size>>>(d_input, d_output, n);

    cudaMemcpy(output.data(), d_output, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
