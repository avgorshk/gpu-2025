#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
} while (0)

#define CHECK_CUFFT(call) do { \
    cufftResult res = call; \
    if (res != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuFFT error"); \
    } \
} while (0)

__global__ void scaleKernel(cufftComplex* data, float factor, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x *= factor;
        data[idx].y *= factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (batch <= 0 || input.size() % (2 * batch) != 0)
        throw std::invalid_argument("Invalid input or batch size");

    const int n = input.size() / (2 * batch);
    const int total = n * batch;
    const size_t bytes = total * sizeof(cufftComplex);

    cufftComplex* d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, batch));

    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scaleKernel<<<blocks, threads>>>(d_data, 1.0f / static_cast<float>(n), total);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> result(input.size());
    CHECK_CUDA(cudaMemcpy(result.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    cudaFree(d_data);

    return result;
}
