#include "fft_cufft.h"
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    const int n = input.size() / (2 * batch);
    const size_t size = input.size() * sizeof(float);
    
    std::vector<float> output(input.size());
    
    cufftComplex* d_data = nullptr;
    cudaMalloc(&d_data, size);
    
    cudaMemcpy(d_data, input.data(), size, cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    cudaMemcpy(output.data(), d_data, size, cudaMemcpyDeviceToHost);
    
    const float norm_factor = 1.0f / n;
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] *= norm_factor;
    }
    
    cufftDestroy(plan);
    cudaFree(d_data);
    
    return output;
}