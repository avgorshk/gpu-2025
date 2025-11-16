#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int BLOCK_SIZE = 256;

__global__ void normalizeKernel(cufftComplex* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    std::vector<float> output(input.size());
    int n = input.size() / (2 * batch);
    int totalElements = n * batch;
    size_t bytes = totalElements * sizeof(cufftComplex);

    cufftComplex* data;
    cudaMalloc(&data, bytes);

    cudaMemcpy(data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    cufftExecC2C(plan, data, data, CUFFT_FORWARD);
    cufftExecC2C(plan, data, data, CUFFT_INVERSE);

    float scale = 1.0f / n;
    int numBlocks = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    normalizeKernel<<<numBlocks, BLOCK_SIZE>>>(data, scale, totalElements);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), data, bytes, cudaMemcpyDeviceToHost);
    cufftDestroy(plan);
    cudaFree(data);

    return output;
}