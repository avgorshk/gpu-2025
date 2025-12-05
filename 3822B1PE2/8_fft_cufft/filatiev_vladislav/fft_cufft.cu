#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

__global__ void normalize_kernel(cufftComplex* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= factor;
        data[idx].y *= factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.empty() || batch <= 0) {
        throw std::invalid_argument("invalid input");
    }

    int n = input.size() / (2 * batch);
    size_t total = n * batch;
    size_t size = total * sizeof(cufftComplex);

    std::vector<float> result(input.size());

    cufftComplex* d_data;
    cudaMalloc(&d_data, size);

    cudaMemcpy(d_data, input.data(), size, cudaMemcpyHostToDevice);

    cufftHandle plan_forward, plan_inverse;
    cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch);
    cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch);

    cufftExecC2C(plan_forward, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan_inverse, d_data, d_data, CUFFT_INVERSE);

    float norm = 1.0f / n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    normalize_kernel << <blocks, threads >> > (d_data, norm, total);

    cudaMemcpy(result.data(), d_data, size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cudaFree(d_data);

    return result;
}