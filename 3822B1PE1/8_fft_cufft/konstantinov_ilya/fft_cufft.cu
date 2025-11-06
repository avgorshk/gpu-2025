#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void normalizeKernel(cufftComplex* data, float norm, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x /= norm;
        data[idx].y /= norm;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int total = input.size() / 2;
    int n = total / batch;
    size_t complexSize = sizeof(cufftComplex) * total;
    cufftComplex* d_data;
    cudaMalloc(&d_data, complexSize);
    std::vector<cufftComplex> h_data(total);
    for (int i = 0; i < total; ++i) {
        h_data[i].x = input[2 * i];
        h_data[i].y = input[2 * i + 1];
    }
    cudaMemcpy(d_data, h_data.data(), complexSize, cudaMemcpyHostToDevice);
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    normalizeKernel<<<blocks, threads>>>(d_data, static_cast<float>(n), total);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data.data(), d_data, complexSize, cudaMemcpyDeviceToHost);
    cufftDestroy(plan);
    cudaFree(d_data);
    std::vector<float> output(input.size());
    for (int i = 0; i < total; ++i) {
        output[2 * i]     = h_data[i].x;
        output[2 * i + 1] = h_data[i].y;
    }

    return output;
}
