#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

__global__ void normalize(cufftComplex* data, int n, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * batch;
    if (idx < total) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("Input size must be 2 * n * batch");
    }

    int n = static_cast<int>(input.size() / (2 * batch));
    size_t complex_size = sizeof(cufftComplex);
    size_t total_bytes = n * batch * complex_size;

    cufftComplex* d_data = nullptr;
    cudaMalloc(&d_data, total_bytes);

    std::vector<cufftComplex> host_data(n * batch);
    for (int i = 0; i < n * batch; ++i) {
        host_data[i].x = input[2 * i];
        host_data[i].y = input[2 * i + 1];
    }

    cudaMemcpy(d_data, host_data.data(), total_bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        cudaFree(d_data);
        throw std::runtime_error("cufftPlan1d failed");
    }

    if (cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("cufftExecC2C (forward) failed");
    }

    if (cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("cufftExecC2C (inverse) failed");
    }

    int threads = 256;
    int blocks = (n * batch + threads - 1) / threads;
    normalize<<<blocks, threads>>>(d_data, n, batch);
    cudaDeviceSynchronize();

    cudaMemcpy(host_data.data(), d_data, total_bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    std::vector<float> result(2 * n * batch);
    for (int i = 0; i < n * batch; ++i) {
        result[2 * i]     = host_data[i].x;
        result[2 * i + 1] = host_data[i].y;
    }

    return result;
}
