#include "gelu_cuda.h"
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

__device__ float gelu_exp_formula(float x) {
    return 0.5f * x * (1.0f + expf(-0.5f * x * x));  // Формула GELU с экспоненцией
}

__global__ void gelu_kernel(float* output, const float* input, size_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Индекс текущего потока
    if (i < len) {
        output[i] = gelu_exp_formula(input[i]);  // Применение GELU с экспонентой
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t len = input.size();
    std::vector<float> output(len);  // Результат вычисления

    float* d_input;
    float* d_output;

    cudaMalloc((void**)&d_input, len * sizeof(float));
    cudaMalloc((void**)&d_output, len * sizeof(float));

    cudaMemcpy(d_input, input.data(), len * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;  // Количество потоков в блоке
    int numBlocks = (len + blockSize - 1) / blockSize;  // Количество блоков

    gelu_kernel<<<numBlocks, blockSize>>>(d_output, d_input, len);

    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
