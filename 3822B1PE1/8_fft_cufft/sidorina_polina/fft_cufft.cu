#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

__global__ void NormalizeKernel(cufftComplex* complex_data, int total_elements, float scale_factor)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        complex_data[idx].x *= scale_factor;
        complex_data[idx].y *= scale_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch_size)
{
    const int fft_size = static_cast<int>(input.size() / (2 * batch_size));
    const int total_complex_elements = fft_size * batch_size;
    const size_t data_size_bytes = input.size() * sizeof(float);
    
    std::vector<float> result(input.size());
    
    cufftComplex* d_complex_data = nullptr;
    cudaMalloc(&d_complex_data, data_size_bytes);
    
    cudaMemcpy(d_complex_data, input.data(), data_size_bytes, cudaMemcpyHostToDevice);
    
    cufftHandle fft_plan;
    cufftPlan1d(&fft_plan, fft_size, CUFFT_C2C, batch_size);
    
    cufftExecC2C(fft_plan, d_complex_data, d_complex_data, CUFFT_FORWARD);
    
    cufftExecC2C(fft_plan, d_complex_data, d_complex_data, CUFFT_INVERSE);
    
    constexpr int THREADS_PER_BLOCK = 256;
    const int blocks_per_grid = (total_complex_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const float normalization_factor = 1.0f / static_cast<float>(fft_size);
    
    NormalizeKernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(
        d_complex_data, total_complex_elements, normalization_factor);
    
    cudaMemcpy(result.data(), d_complex_data, data_size_bytes, cudaMemcpyDeviceToHost);
    
    cufftDestroy(fft_plan);
    cudaFree(d_complex_data);
    
    return result;
}