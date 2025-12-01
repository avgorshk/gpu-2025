#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

constexpr float GELU_SCALE = 0.7978845608028654f;
constexpr float GELU_PARAM = 0.044715f;
constexpr float HALF_VAL = 0.5f;

__global__ void gelu_kernel(const float* __restrict__ input_data,
    float* __restrict__ output_data,
    int data_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < data_size) {
        float x = input_data[idx];
        float x_cubed = x * x * x;
        float transformed = GELU_SCALE * (x + GELU_PARAM * x_cubed);
        float exp_val = expf(2.0f * transformed);
        float tanh_approx = (exp_val - 1.0f) / (exp_val + 1.0f);
        output_data[idx] = HALF_VAL * x * (1.0f + tanh_approx);
    }
}

void checkCudaError(cudaError_t err_code, const char* err_msg) {
    if (err_code != cudaSuccess) {
        std::cerr << "CUDA Error - " << err_msg << ": " << cudaGetErrorString(err_code) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err_code));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int num_elements = input.size();
    std::vector<float> result(num_elements);

    if (num_elements == 0) {
        return result;
    }

    float* d_input = nullptr;
    float* d_output = nullptr;

    checkCudaError(cudaMalloc(&d_input, num_elements * sizeof(float)),
        "Device input allocation failed");
    checkCudaError(cudaMalloc(&d_output, num_elements * sizeof(float)),
        "Device output allocation failed");

    checkCudaError(cudaMemcpy(d_input, input.data(), num_elements * sizeof(float),
        cudaMemcpyHostToDevice), "Input copy to device failed");

    const int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    gelu_kernel << <grid_size, block_size >> > (d_input, d_output, num_elements);

    checkCudaError(cudaGetLastError(), "Kernel execution failed");
    checkCudaError(cudaDeviceSynchronize(), "Device synchronization failed");
    checkCudaError(cudaMemcpy(result.data(), d_output, num_elements * sizeof(float),
        cudaMemcpyDeviceToHost), "Output copy from device failed");

    checkCudaError(cudaFree(d_input), "Device input free failed");
    checkCudaError(cudaFree(d_output), "Device output free failed");

    return result;
}