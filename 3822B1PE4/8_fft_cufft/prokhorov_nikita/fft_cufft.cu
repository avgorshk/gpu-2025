#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void normalizeKernel(cufftComplex* data, int total_elements, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    
    cufftComplex* d_data;
    size_t size = n * batch * sizeof(cufftComplex);
    cudaMalloc(&d_data, size);
    
    cudaMemcpy(d_data, input.data(), size, cudaMemcpyHostToDevice);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    float scale = 1.0f / n;
    int total_elements = n * batch;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    normalizeKernel<<<numBlocks, blockSize>>>(d_data, total_elements, scale);
    
    cudaDeviceSynchronize();
    
    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), d_data, size, cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(d_data);
    
    return output;
}