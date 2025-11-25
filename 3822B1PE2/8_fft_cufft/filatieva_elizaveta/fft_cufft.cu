#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>

__global__ void normalize_complex_data(cufftComplex* complex_array, float normalization_factor, int total_elements) {
    int element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index < total_elements) {
        complex_array[element_index].x *= normalization_factor;
        complex_array[element_index].y *= normalization_factor;
    }
}

void checkCudaError(cudaError_t error_code, const char* error_message) {
    if (error_code != cudaSuccess) {
        std::cerr << "CUDA Error (" << error_message << "): "
            << cudaGetErrorString(error_code) << std::endl;
        throw std::runtime_error(cudaGetErrorString(error_code));
    }
}

void checkCufftError(cufftResult_t result_code, const char* operation_name) {
    if (result_code != CUFFT_SUCCESS) {
        std::cerr << "cuFFT Error (" << operation_name << "): " << result_code << std::endl;
        throw std::runtime_error("cuFFT operation failed");
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input_signal, int batch_count) {
    if (input_signal.empty()) {
        return std::vector<float>();
    }
    if (batch_count <= 0) {
        throw std::invalid_argument("Batch count must be positive");
    }
    if (input_signal.size() % (2 * batch_count) != 0) {
        throw std::invalid_argument("Input size must be divisible by 2 * batch");
    }

    int signal_length = input_signal.size() / (2 * batch_count);
    if (signal_length <= 0) {
        throw std::invalid_argument("Invalid signal length");
    }

    size_t total_complex_elements = signal_length * batch_count;
    size_t memory_bytes = total_complex_elements * sizeof(cufftComplex);

    std::vector<float> result_signal(input_signal.size());

    cufftComplex* device_input = nullptr;
    cufftComplex* device_fft_result = nullptr;

    checkCudaError(cudaMalloc(&device_input, memory_bytes), "Allocating device memory for input");
    checkCudaError(cudaMalloc(&device_fft_result, memory_bytes), "Allocating device memory for FFT result");

    cudaStream_t computation_stream;
    checkCudaError(cudaStreamCreate(&computation_stream), "Creating CUDA stream");

    checkCudaError(cudaMemcpyAsync(device_input, input_signal.data(), memory_bytes,
        cudaMemcpyHostToDevice, computation_stream),
        "Copying input data to device");

    cufftHandle forward_plan;
    checkCufftError(cufftPlan1d(&forward_plan, signal_length, CUFFT_C2C, batch_count),
        "Creating forward FFT plan");

    cufftHandle inverse_plan;
    checkCufftError(cufftPlan1d(&inverse_plan, signal_length, CUFFT_C2C, batch_count),
        "Creating inverse FFT plan");

    checkCufftError(cufftSetStream(forward_plan, computation_stream), "Setting stream for forward FFT");
    checkCufftError(cufftSetStream(inverse_plan, computation_stream), "Setting stream for inverse FFT");

    checkCufftError(cufftExecC2C(forward_plan, device_input, device_fft_result, CUFFT_FORWARD),
        "Executing forward FFT");

    checkCufftError(cufftExecC2C(inverse_plan, device_fft_result, device_input, CUFFT_INVERSE),
        "Executing inverse FFT");

    float normalization_factor = 1.0f / signal_length;
    const int threads_per_block = 256;
    int block_count = (total_complex_elements + threads_per_block - 1) / threads_per_block;

    normalize_complex_data << <block_count, threads_per_block, 0, computation_stream >> > (
        device_input, normalization_factor, total_complex_elements);

    checkCudaError(cudaGetLastError(), "Normalization kernel execution");

    checkCudaError(cudaMemcpyAsync(result_signal.data(), device_input, memory_bytes,
        cudaMemcpyDeviceToHost, computation_stream),
        "Copying result from device");

    checkCudaError(cudaStreamSynchronize(computation_stream), "Synchronizing stream");

    checkCufftError(cufftDestroy(forward_plan), "Destroying forward FFT plan");
    checkCufftError(cufftDestroy(inverse_plan), "Destroying inverse FFT plan");
    checkCudaError(cudaStreamDestroy(computation_stream), "Destroying CUDA stream");

    checkCudaError(cudaFree(device_input), "Freeing device input memory");
    checkCudaError(cudaFree(device_fft_result), "Freeing device FFT result memory");

    return result_signal;
}