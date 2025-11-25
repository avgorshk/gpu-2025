
#include "gelu_cuda.h"

#include <cmath>
#include <iostream>
#include <vector>

__global__ void kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float tanh_result = tanhf(sqrtf(2.0f / M_PI) * input[idx] * (1 + 0.044715f * powf(input[idx],2)));
        output[idx] = 0.5f * input[idx] * (1.0f + tanh_result);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    float* cuda_input;
    float* cuda_output;
    cudaMalloc((void**)&cuda_input, input.size() * sizeof(float));
    cudaMalloc((void**)&cuda_output, input.size() * sizeof(float));
    cudaMemcpy(cuda_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    size_t blockSize = 256;
    size_t numBlocks = (input.size() + blockSize - 1) / blockSize;
    kernel<<<numBlocks, blockSize>>>(cuda_input, cuda_output, input.size());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout<<"Kernel launch error: "<<cudaGetErrorString(err)<<std::endl;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), cuda_output, input.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cuda_input);
    cudaFree(cuda_output);
    return output;
}
