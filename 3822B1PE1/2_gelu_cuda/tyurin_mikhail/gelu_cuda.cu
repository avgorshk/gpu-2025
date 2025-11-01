#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <stdexcept>

struct GeluConstants {
    float k0 = 0.7978845608028654f;
    float k1 = 0.044715f;
};

__global__ void gelu_kernel(const float* input, float* output, size_t n, GeluConstants c) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float inner = c.k0 * (x + c.k1 * x * x * x);

        float e = __expf(2.0f * inner);
        float tanh_approx = (e - 1.0f) / (e + 1.0f);

        output[idx] = 0.5f * x * (1.0f + tanh_approx);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    GeluConstants c;
    size_t blockSize = 256;
    size_t numBlocks = (n + blockSize - 1) / blockSize;
    gelu_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n, c);

    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
