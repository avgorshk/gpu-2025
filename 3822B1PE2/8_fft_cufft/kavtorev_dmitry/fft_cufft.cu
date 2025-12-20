#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void normalize_kernel(cufftComplex* data, float norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= norm;
        data[idx].y *= norm;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    
    size_t complex_size = n * batch * sizeof(cufftComplex);
    std::vector<float> output(input.size());
    
    cufftComplex *d_data;
    cudaMalloc(&d_data, complex_size);
    
    cudaMemcpy(d_data, input.data(), complex_size, cudaMemcpyHostToDevice);
    
    cufftHandle plan_forward;
    cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch);
    
    cufftExecC2C(plan_forward, d_data, d_data, CUFFT_FORWARD);
    
    cufftHandle plan_inverse;
    cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch);
    
    cufftExecC2C(plan_inverse, d_data, d_data, CUFFT_INVERSE);
    
    float norm = 1.0f / static_cast<float>(n);
    int total_elements = n * batch;
    
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    normalize_kernel<<<numBlocks, blockSize>>>(d_data, norm, total_elements);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data(), d_data, complex_size, cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cudaFree(d_data);
    
    return output;
}

