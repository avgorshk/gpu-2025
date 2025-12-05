#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void GeluKernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = input[i];
        float x3 = x * x * x;

        float y = 0.79788456f * (x + 0.044715f * x3); 
        float exp_val = expf(y);
        output[i] = x * exp_val / (exp_val + 1.0f);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();
    std::vector<float> output(size);

    if (size == 0)
        return output;

    float *d_input = nullptr;
    float *d_output = nullptr;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpyAsync(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    GeluKernel<<<grid_size, block_size>>>(d_input, d_output, size);

    cudaMemcpyAsync(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
