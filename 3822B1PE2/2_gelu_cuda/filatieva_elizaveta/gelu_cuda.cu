#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

constexpr float SCALE_FACTOR = 0.7978845608028654f;
constexpr float CUBE_COEFF = 0.044715f;
constexpr float HALF = 0.5f;

__global__ void compute_gelu(const float* __restrict__ input,
    float* __restrict__ output,
    int total_elements) {
    int element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_idx < total_elements) {
        float value = input[element_idx];
        float value_cubed = value * value * value;
        float transformed = SCALE_FACTOR * (value + CUBE_COEFF * value_cubed);
        float exp_val = expf(2.0f * transformed);
        float tanh_approximation = (exp_val - 1.0f) / (exp_val + 1.0f);

        output[element_idx] = HALF * value * (1.0f + tanh_approximation);
    }
}

inline void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int element_count = input.size();
    std::vector<float> result(element_count);

    if (element_count == 0) {
        return result;
    }

    float* device_input = nullptr;
    float* device_output = nullptr;

    checkCudaError(cudaMalloc(&device_input, element_count * sizeof(float)),
        "Failed to allocate device input memory");
    checkCudaError(cudaMalloc(&device_output, element_count * sizeof(float)),
        "Failed to allocate device output memory");

    checkCudaError(cudaMemcpy(device_input, input.data(), element_count * sizeof(float),
        cudaMemcpyHostToDevice),
        "Failed to copy input to device");
    const int threads_per_block = 256;
    int blocks_per_grid = (element_count + threads_per_block - 1) / threads_per_block;

    compute_gelu << <blocks_per_grid, threads_per_block >> > (device_input, device_output, element_count);

    checkCudaError(cudaGetLastError(), "Kernel execution failed");
    checkCudaError(cudaDeviceSynchronize(), "Device synchronization failed");
    checkCudaError(cudaMemcpy(result.data(), device_output, element_count * sizeof(float),
        cudaMemcpyDeviceToHost),
        "Failed to copy output from device");

    checkCudaError(cudaFree(device_input), "Failed to free device input memory");
    checkCudaError(cudaFree(device_output), "Failed to free device output memory");

    return result;
}