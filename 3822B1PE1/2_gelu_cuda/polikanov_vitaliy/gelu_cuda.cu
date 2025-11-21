#include "cuda_runtime.h"
#include "gelu_cuda.h"
#include <vector>
#include <chrono>


__global__ void tkernel(float *res, const float *input, const size_t size) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        const float x = input[i];
        const float arg = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        const float tanh_ = tanh(arg);
        res[i] = 0.5f * x * (1.0f + tanh_);
    }
}

std::vector<float> GeluCUDA(const std::vector<float> &input_data) {
    const size_t data_count = input_data.size();
    std::vector<float> output_result(data_count);

    float *device_input, *device_output;

    cudaMalloc((void **) &device_input, data_count * sizeof(float));
    cudaMalloc((void **) &device_output, data_count * sizeof(float));

    cudaMemcpy(device_input, input_data.data(), data_count * sizeof(float), cudaMemcpyHostToDevice);

    const int threads_per_block = 128;
    const int block_count = (data_count + threads_per_block - 1) / threads_per_block;

    tkernel <<<block_count, threads_per_block>>>(device_output, device_input, data_count);

    cudaDeviceSynchronize();
    cudaMemcpy(output_result.data(), device_output, data_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_input);
    cudaFree(device_output);
    return output_result;
}