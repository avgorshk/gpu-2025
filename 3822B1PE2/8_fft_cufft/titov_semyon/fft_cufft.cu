#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>

__global__ void normalize_kernel(cufftComplex* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

void checkCufftError(cufftResult_t result, const char* msg) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT error: " << msg << " - " << result << std::endl;
        throw std::runtime_error("cuFFT error");
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.empty()) {
        return std::vector<float>();
    }
    if (batch <= 0) {
        throw std::invalid_argument("Batch must be positive");
    }
    if (input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("Input size must be divisible by 2 * batch");
    }
    int n = input.size() / (2 * batch);
    if (n <= 0) {
        throw std::invalid_argument("Invalid signal length");
    }

    size_t complex_size = n * batch;
    size_t bytes = complex_size * sizeof(cufftComplex);

    std::vector<float> output(input.size());

    cufftComplex* d_input = nullptr, * d_fft = nullptr, * d_ifft = nullptr;

    checkCudaError(cudaMalloc(&d_input, bytes), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_fft, bytes), "cudaMalloc d_fft");
    checkCudaError(cudaMalloc(&d_ifft, bytes), "cudaMalloc d_ifft");
    checkCudaError(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "Copy input to device");
    cufftHandle plan_forward;
    checkCufftError(cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch), "cufftPlan1d forward");

    cufftHandle plan_inverse;
    checkCufftError(cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch), "cufftPlan1d inverse");
    checkCufftError(cufftExecC2C(plan_forward, d_input, d_fft, CUFFT_FORWARD), "cufftExecC2C forward");
    checkCufftError(cufftExecC2C(plan_inverse, d_fft, d_ifft, CUFFT_INVERSE), "cufftExecC2C inverse");

    float scale = 1.0f / n;
    int total_elements = complex_size;

    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    normalize_kernel << <numBlocks, blockSize >> > (d_ifft, scale, total_elements);

    checkCudaError(cudaGetLastError(), "Normalization kernel");
    checkCudaError(cudaDeviceSynchronize(), "Kernel synchronization");

    checkCudaError(cudaMemcpy(output.data(), d_ifft, bytes, cudaMemcpyDeviceToHost), "Copy result from device");

    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cudaFree(d_input);
    cudaFree(d_fft);
    cudaFree(d_ifft);

    return output;
}