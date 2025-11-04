#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <iostream>


__global__ void normalizeKernel(cufftComplex* device_data, int total_complex_elements, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_complex_elements) {
        device_data[idx].x *= scale_factor;
        device_data[idx].y *= scale_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input_data, int batch_size) {

    int signal_length = input_data.size() / (2 * batch_size);
    int total_complex_elements = signal_length * batch_size;
    
    if (signal_length <= 0) {
        return std::vector<float>();
    }

    cufftComplex *device_data = nullptr;
    size_t data_size_bytes = total_complex_elements * sizeof(cufftComplex);

    cudaMalloc(reinterpret_cast<void**>(&device_data), data_size_bytes);
    cudaMemcpy(device_data, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    cufftHandle plan_handle;
    cufftPlan1d(&plan_handle, signal_length, CUFFT_C2C, batch_size);

    std::vector<float> host_result(input_data.size());

    cufftExecC2C(plan_handle, device_data, device_data, CUFFT_FORWARD);
    cufftExecC2C(plan_handle, device_data, device_data, CUFFT_INVERSE);

    const int optimal_block_size = 256;
    int num_blocks = (total_complex_elements + optimal_block_size - 1) / optimal_block_size;
    
    const float scale_factor = 1.0f / static_cast<float>(signal_length);

    normalizeKernel<<<num_blocks, optimal_block_size>>>(
        device_data, 
        total_complex_elements, 
        scale_factor
    );

    cudaMemcpy(host_result.data(), device_data, input_data.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cufftDestroy(plan_handle);
    cudaFree(device_data);
    
    return host_result;
}