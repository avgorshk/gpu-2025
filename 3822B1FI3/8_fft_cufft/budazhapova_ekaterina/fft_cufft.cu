#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void normalize_kernel(cufftComplex* data, int n, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FftCUFFT(const std::vector<float>& input, int batch) {
    if (input.size() % (2 * batch) != 0) {
        std::cerr << "Error: Input size must be divisible by 2*batch" << std::endl;
        return std::vector<float>();
    }
    
    int n = input.size() / (2 * batch); 
    size_t total_complex = input.size() / 2;
    size_t bytes = input.size() * sizeof(float);
    
    cufftComplex* d_data;
    cudaMalloc(&d_data, bytes);
    
    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    int threads = 256;
    int blocks = (total_complex + threads - 1) / threads;
    float norm_factor = 1.0f / n;
    normalize_kernel<<<blocks, threads>>>(d_data, n, total_complex);
    
    std::vector<float> result(input.size());
    cudaMemcpy(result.data(), d_data, bytes, cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(d_data);
    
    return result;
}