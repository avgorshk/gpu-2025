#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("Input size must be divisible by 2*batch");
    }
    
    size_t n = input.size() / (2 * batch);
    size_t complex_size = n * batch;
    size_t float_size = 2 * complex_size * sizeof(float);
    
    cufftComplex *d_data;
    cudaMalloc(&d_data, float_size);
    cudaMemcpy(d_data, input.data(), float_size, cudaMemcpyHostToDevice);
    
    cufftHandle plan_forward, plan_inverse;
    cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch);
    cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch);
    
    cufftExecC2C(plan_forward, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan_inverse, d_data, d_data, CUFFT_INVERSE);
    
    float scale = 1.0f / n;
    cufftComplex *h_result = new cufftComplex[complex_size];
    
    cudaMemcpy(h_result, d_data, float_size, cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < complex_size; ++i) {
        h_result[i].x *= scale;
        h_result[i].y *= scale;
    }
    
    std::vector<float> result(2 * complex_size);
    for (size_t i = 0; i < complex_size; ++i) {
        result[2 * i] = h_result[i].x;
        result[2 * i + 1] = h_result[i].y;
    }
    
    delete[] h_result;
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cudaFree(d_data);
    
    return result;
}