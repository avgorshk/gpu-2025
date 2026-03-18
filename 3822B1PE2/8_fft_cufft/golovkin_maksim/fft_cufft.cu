#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void NormalizeKernel(cufftComplex* data, int n, int totalSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSize) {
        data[idx].x /= static_cast<float>(n);
        data[idx].y /= static_cast<float>(n);
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = static_cast<int>(input.size()) / (2 * batch);
    int totalSize = n * batch;

    std::vector<float> output(input.size());

    cufftComplex* d_data = nullptr;
    cudaMalloc(&d_data, totalSize * sizeof(cufftComplex));
    cudaMemcpy(d_data, input.data(), totalSize * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    const int kBlockSize = 256;
    int numBlocks = (totalSize + kBlockSize - 1) / kBlockSize;
    NormalizeKernel<<<numBlocks, kBlockSize>>>(d_data, n, totalSize);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_data, totalSize * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}
