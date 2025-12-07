#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void applyNormalization(cufftComplex* buffer, int num_elements, float factor) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx < num_elements) {
        buffer[global_idx].x = buffer[global_idx].x * factor;
        buffer[global_idx].y = buffer[global_idx].y * factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int signal_length = input.size() / (2 * batch);
    int complex_count = signal_length * batch;
    
    static cufftComplex* device_buffer = nullptr;
    static size_t current_capacity = 0;
    static cufftHandle fft_plan = 0;
    static int cached_signal_length = 0;
    static int cached_batch = 0;
    
    size_t required_bytes = complex_count * sizeof(cufftComplex);
    
    if (current_capacity < required_bytes) {
        if (device_buffer != nullptr) {
            cudaFree(device_buffer);
        }
        cudaMalloc((void**)&device_buffer, required_bytes);
        current_capacity = required_bytes;
    }
    
    if (fft_plan == 0 || cached_signal_length != signal_length || cached_batch != batch) {
        if (fft_plan != 0) {
            cufftDestroy(fft_plan);
        }
        cufftPlan1d(&fft_plan, signal_length, CUFFT_C2C, batch);
        cached_signal_length = signal_length;
        cached_batch = batch;
    }
    
    cudaMemcpy(device_buffer, input.data(), required_bytes, cudaMemcpyHostToDevice);
    
    cufftExecC2C(fft_plan, device_buffer, device_buffer, CUFFT_FORWARD);
    cufftExecC2C(fft_plan, device_buffer, device_buffer, CUFFT_INVERSE);
    
    const int block_size = 256;
    int grid_size = (complex_count + block_size - 1) / block_size;
    float normalization_factor = 1.0f / static_cast<float>(signal_length);
    applyNormalization<<<grid_size, block_size>>>(device_buffer, complex_count, normalization_factor);
    
    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), device_buffer, required_bytes, cudaMemcpyDeviceToHost);
    
    return output;
}
