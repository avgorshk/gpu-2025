#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void normalize_kernel(cufftComplex* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    int total_complex = n * batch;
    
    std::vector<float> output(input.size());
    
    cufftComplex* d_data;
    size_t bytes = total_complex * sizeof(cufftComplex);
    cudaMalloc(&d_data, bytes);
    
    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    float scale = 1.0f / n;
    int blockSize = 256;
    int gridSize = (total_complex + blockSize - 1) / blockSize;
    normalize_kernel<<<gridSize, blockSize>>>(d_data, total_complex, scale);
    
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(d_data);
    
    return output;
}