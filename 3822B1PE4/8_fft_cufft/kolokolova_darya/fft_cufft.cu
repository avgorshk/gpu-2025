#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <stdexcept>

__global__ void NormalizeComplexKernel(cufftComplex* complex_data, int total_elements, float normalization_factor) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        complex_data[index].x *= normalization_factor;
        complex_data[index].y *= normalization_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.empty() || batch <= 0) {
        throw std::invalid_argument("Invalid input: empty input or non-positive batch size");
    }
    
    if (input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("Invalid input size: must be divisible by 2 * batch");
    }

    const int signal_length = input.size() / (2 * batch);
    const int total_complex_elements = signal_length * batch;
    const size_t data_bytes = sizeof(cufftComplex) * total_complex_elements;

    // Allocate device memory
    cufftComplex* device_data = nullptr;
    cudaError_t cuda_status = cudaMalloc(&device_data, data_bytes);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
    }

    // Copy input data to device
    cuda_status = cudaMemcpy(device_data, input.data(), data_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(device_data);
        throw std::runtime_error("Failed to copy data to device");
    }

    // Create FFT plan
    cufftHandle fft_plan;
    cufftResult cufft_status = cufftPlan1d(&fft_plan, signal_length, CUFFT_C2C, batch);
    if (cufft_status != CUFFT_SUCCESS) {
        cudaFree(device_data);
        throw std::runtime_error("Failed to create FFT plan");
    }

    try {
        // Execute forward FFT
        cufft_status = cufftExecC2C(fft_plan, device_data, device_data, CUFFT_FORWARD);
        if (cufft_status != CUFFT_SUCCESS) {
            throw std::runtime_error("Forward FFT execution failed");
        }

        // Execute inverse FFT
        cufft_status = cufftExecC2C(fft_plan, device_data, device_data, CUFFT_INVERSE);
        if (cufft_status != CUFFT_SUCCESS) {
            throw std::runtime_error("Inverse FFT execution failed");
        }

        // Normalize the result
        const float normalization_factor = 1.0f / static_cast<float>(signal_length);
        const int block_size = 256;
        const int grid_size = (total_complex_elements + block_size - 1) / block_size;
        
        NormalizeComplexKernel<<<grid_size, block_size>>>(device_data, total_complex_elements, normalization_factor);
        
        // Check for kernel launch errors
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }

        // Synchronize to ensure normalization is complete
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        // Copy results back to host
        std::vector<float> output(input.size());
        cuda_status = cudaMemcpy(output.data(), device_data, data_bytes, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy data from device");
        }

        // Cleanup
        cufftDestroy(fft_plan);
        cudaFree(device_data);

        return output;

    } catch (const std::exception& e) {
        // Cleanup in case of error
        cufftDestroy(fft_plan);
        cudaFree(device_data);
        throw;
    }
}