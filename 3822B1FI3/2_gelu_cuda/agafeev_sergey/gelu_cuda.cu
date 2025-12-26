#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SQRT_2_OVER_PI 0.7978845608028654

__global__ void kernel(float *input, float *output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = input[i];
        output[i] = 0.5 * x * (1.0 + std::tanh(SQRT_2_OVER_PI * (x + 0.044715 * (x * x * x))));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t input_sz = input.size();
    const size_t size_for_buf_in_bytes = input_sz * sizeof(float);
    std::vector<float> output(input_sz);

    float *gpu_input;
    float *gpu_output;

    cudaMalloc(&gpu_input, size_for_buf_in_bytes);
    cudaMalloc(&gpu_output, size_for_buf_in_bytes);

    cudaMemcpy(gpu_input, input.data(), size_for_buf_in_bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    int num_blocks = (input_sz + block_size - 1) / block_size;
    kernel<<<num_blocks, block_size>>>(gpu_input, gpu_output, input_sz);

    cudaMemcpy(output.data(), gpu_output, size_for_buf_in_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_input);
    cudaFree(gpu_output);

    return output;
}