#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    
    cufftHandle plan_forward;
    cufftHandle plan_inverse;
    
    cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch);
    cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch);
    
    cufftComplex* d_data = nullptr;
    size_t size = batch * n * sizeof(cufftComplex);
    
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, (cufftComplex*)input.data(), size, cudaMemcpyHostToDevice);
    
    cufftExecC2C(plan_forward, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan_inverse, d_data, d_data, CUFFT_INVERSE);
    
    float scale = 1.0f / n;
    
    int threads = 256;
    int blocks = (batch * n + threads - 1) / threads;
    
    cudaMemcpy(d_data, d_data, size, cudaMemcpyDeviceToDevice);
    
    std::vector<float> result(input.size());
    cudaMemcpy(result.data(), d_data, size, cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] *= scale;
    }
    
    cudaFree(d_data);
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    
    return result;
}