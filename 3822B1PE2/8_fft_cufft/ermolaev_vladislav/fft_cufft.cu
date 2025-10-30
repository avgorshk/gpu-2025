#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>

__global__ void normalizeKernel(cufftComplex* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    int total_elements = n * batch;
    
    std::vector<float> output(input.size());
    
    cufftComplex *d_data;
    size_t size = total_elements * sizeof(cufftComplex);
    cudaMalloc(&d_data, size);

    cudaMemcpy(d_data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, normalizeKernel, 0, 0);
    
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    float scale = 1.0f / n;
    normalizeKernel<<<numBlocks, blockSize>>>(d_data, total_elements, scale);

    cudaMemcpy(output.data(), d_data, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(d_data);
    
    return output;
}