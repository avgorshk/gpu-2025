#include "fft_cufft.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

__global__ void NormalizeKernel(cufftComplex* data, int total, float inv_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x *= inv_n;
        data[idx].y *= inv_n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.empty() || batch <= 0) {
        return std::vector<float>();
    }

    int n = static_cast<int>(input.size() / (2 * batch));
    int total = n * batch;
    size_t bytes = static_cast<size_t>(total) * sizeof(cufftComplex);

    cufftComplex* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    float inv_n = 1.0f / static_cast<float>(n);
    int block = 256;
    int grid = (total + block - 1) / block;
    NormalizeKernel<<<grid, block>>>(d_data, total, inv_n);

    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}
