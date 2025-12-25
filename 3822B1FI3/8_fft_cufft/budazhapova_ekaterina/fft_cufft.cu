#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void normalize_kernel(cufftComplex* data, float norm_factor, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x *= norm_factor;
        data[idx].y *= norm_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    size_t total_complex = n * batch;
    
    cufftComplex* d_data;
    cudaMalloc(&d_data, input.size() * sizeof(float));
    cudaMemcpy(d_data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    int threads = 256;
    int blocks = (total_complex + threads - 1) / threads;
    float norm_factor = 1.0f / static_cast<float>(n);
    normalize_kernel<<<blocks, threads>>>(d_data, norm_factor, total_complex);
    
    std::vector<float> result(input.size());
    cudaMemcpy(result.data(), d_data, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(d_data);
    
    return result;
}