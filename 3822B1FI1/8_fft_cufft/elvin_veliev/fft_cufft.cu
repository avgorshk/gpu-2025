#include "fft_cufft.h"

#include <cufft.h>
#include <cuda_runtime.h>

__global__ void normalize(cufftComplex* data, int n, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * batch;
    if (idx < total) {
        data[idx].x /= n;
	data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    size_t complexSize = sizeof(cufftComplex);
    size_t totalSize = n * batch * complexSize;

    std::vector<float> output(input.size());
    cufftComplex* d_data = nullptr;

    cudaMalloc(&d_data, totalSize);
    cudaMemcpy(d_data, input.data(), totalSize, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int threads = 256;
    int blocks = (n * batch + threads - 1) / threads;
    normalize << <blocks, threads >> > (d_data, n, batch);

    cudaMemcpy(output.data(), d_data, totalSize, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}