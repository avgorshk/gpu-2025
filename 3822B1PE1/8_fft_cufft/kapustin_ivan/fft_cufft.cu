#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void normalizeKernel(cufftComplex* data, int n, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x /= static_cast<float>(n);
        data[idx].y /= static_cast<float>(n);
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch)
{
    if (batch <= 0 || input.empty()) {
        return {};
    }

    const int totalComplex = input.size() / 2;
    const int n = totalComplex / batch;
    const size_t bytes = sizeof(cufftComplex) * totalComplex;

    cufftComplex* devBuf = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&devBuf), bytes);
    cudaMemcpy(devBuf, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, devBuf, devBuf, CUFFT_FORWARD);
    cufftExecC2C(plan, devBuf, devBuf, CUFFT_INVERSE);

    int blockSize = 256;
    int gridSize  = (totalComplex + blockSize - 1) / blockSize;
    normalizeKernel<<<gridSize, blockSize>>>(devBuf, n, totalComplex);

    std::vector<float> result(input.size());
    cudaMemcpy(result.data(), devBuf, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(devBuf);

    return result;
}
