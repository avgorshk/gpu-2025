#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void normalizeKernel(cufftComplex* data, int n, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    int total_complex = n * batch;
    
    size_t bytes = total_complex * sizeof(cufftComplex);
    
    cufftComplex* d_data;
    cudaMalloc(&d_data, bytes);
    
    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    int blockSize = 256;
    int numBlocks = (total_complex + blockSize - 1) / blockSize;
    normalizeKernel<<<numBlocks, blockSize>>>(d_data, n, total_complex);
    
    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(d_data);
    
    return output;
}