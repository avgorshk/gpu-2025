#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include "fft_cufft.h"

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {

    size_t complex_count = input.size() / 2;
    size_t fft_size = complex_count / batch;
    std::vector<float> processed(complex_count * 2);
    cufftComplex* device_buffer = nullptr;
    cudaMalloc((void**)&device_buffer, complex_count * sizeof(cufftComplex));
    cufftHandle fft_plan;
    cufftPlan1d(&fft_plan, fft_size, CUFFT_C2C, batch);
    cudaMemcpy(device_buffer, input.data(),
               complex_count * sizeof(cufftComplex),
               cudaMemcpyHostToDevice);
    cufftExecC2C(fft_plan, device_buffer, device_buffer, CUFFT_FORWARD);
    cufftExecC2C(fft_plan, device_buffer, device_buffer, CUFFT_INVERSE);
    cudaMemcpy(processed.data(), device_buffer,
               complex_count * sizeof(cufftComplex),
               cudaMemcpyDeviceToHost);
    float normalization = 1.0f / fft_size;
    for (size_t idx = 0; idx < processed.size(); ++idx) {
        processed[idx] *= normalization;
    }
    cufftDestroy(fft_plan);
    cudaFree(device_buffer);
    return processed;
}