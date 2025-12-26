#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>

__global__ void normalizeKernel(cufftComplex* data, int size, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    int totalElements = n * batch;
    
    std::vector<float> output(input.size());
    
    cufftComplex *deviceData;
    size_t bytes = totalElements * sizeof(cufftComplex);
    cudaMalloc(&deviceData, bytes);

    cudaMemcpy(deviceData, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    cufftHandle cufftPlan;
    cufftPlan1d(&cufftPlan, n, CUFFT_C2C, batch);
    
    cufftExecC2C(cufftPlan, deviceData, deviceData, CUFFT_FORWARD);
    cufftExecC2C(cufftPlan, deviceData, deviceData, CUFFT_INVERSE);
    
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, normalizeKernel, 0, 0);
    
    int numBlocks = (totalElements + blockSize - 1) / blockSize;
    float scale = 1.0f / n;
    normalizeKernel<<<numBlocks, blockSize>>>(deviceData, totalElements, scale);

    cudaMemcpy(output.data(), deviceData, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    cufftDestroy(cufftPlan);
    cudaFree(deviceData);
    
    return output;
}