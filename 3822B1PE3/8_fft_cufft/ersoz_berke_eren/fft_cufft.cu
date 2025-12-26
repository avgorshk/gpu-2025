#include "fft_cufft.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>

__global__ void normalizeKernel(cufftComplex* data, int totalElements, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.empty() || batch <= 0) {
        return {};
    }

    // Input contains (real, imag) pairs, so size = 2 * n * batch
    int n = static_cast<int>(input.size()) / (2 * batch);
    int totalElements = n * batch;

    std::vector<float> output(input.size());

    cufftComplex* deviceData = nullptr;
    cudaMalloc(&deviceData, totalElements * sizeof(cufftComplex));

    // Copy input to device (cufftComplex has same layout as float pairs)
    cudaMemcpy(deviceData, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuFFT plan for batched 1D FFT
    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        cudaFree(deviceData);
        throw std::runtime_error("cufftPlan1d failed");
    }

    // Execute forward FFT
    if (cufftExecC2C(plan, deviceData, deviceData, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(deviceData);
        throw std::runtime_error("cufftExecC2C forward failed");
    }

    // Execute inverse FFT
    if (cufftExecC2C(plan, deviceData, deviceData, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(deviceData);
        throw std::runtime_error("cufftExecC2C inverse failed");
    }

    // Normalize by n (inverse FFT in cuFFT is not normalized)
    const int blockSize = 256;
    const int gridSize = (totalElements + blockSize - 1) / blockSize;
    float scale = 1.0f / static_cast<float>(n);
    normalizeKernel<<<gridSize, blockSize>>>(deviceData, totalElements, scale);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output.data(), deviceData, input.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(deviceData);

    return output;
}
