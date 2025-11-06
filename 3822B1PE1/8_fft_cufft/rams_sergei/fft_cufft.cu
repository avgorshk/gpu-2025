#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void kernel(cufftComplex* data, float scale, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    int data_size = n * batch * sizeof(cufftComplex);
    int total = n * batch;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    cufftComplex* data;
    cudaMalloc(&data, data_size);
    cudaMemcpy(data, input.data(), data_size, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, data, data, CUFFT_FORWARD);
    cufftExecC2C(plan, data, data, CUFFT_INVERSE);

    kernel<<<grid_size, block_size>>>(data, 1.0f / static_cast<float>(n), total);

    std::vector<float> result(input.size());
    cudaMemcpy(result.data(), data, data_size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(data);

    return result;
}

