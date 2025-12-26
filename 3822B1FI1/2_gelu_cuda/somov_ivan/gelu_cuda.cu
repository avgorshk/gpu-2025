#include "gelu_cuda.h"

#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const float sq = 0.797884f;

__global__ void geluKernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(sq * x * (1.0f + 0.044715f * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t size = input.size();
    size_t bytes = size * sizeof(float);

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t blocksize = deviceProp.maxThreadsPerBlock;
    size_t gridsize = (size + blocksize - 1) / blocksize;

    geluKernel<<<gridsize, blocksize>>>(d_input, d_output, size);

    std::vector<float> output(size);
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}