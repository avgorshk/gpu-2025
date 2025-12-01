#include "fft_cufft.h"
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

__global__ void normalize_kernel(cufftComplex* data, float factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx].x *= factor;  
    data[idx].y *= factor; 
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);  
    cufftComplex* d_data = nullptr;
    cudaMalloc(&d_data, sizeof(cufftComplex) * n * batch); 

    cudaMemcpy(d_data, input.data(), sizeof(float) * 2 * n * batch, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);


    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    float normalization_factor = 1.0f / n;
    int total_elements = 2 * n * batch;

  
    int threads_per_block = 256;  
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;  

    normalize_kernel<<<num_blocks, threads_per_block>>>(d_data, normalization_factor, total_elements);

    cudaDeviceSynchronize();  

    std::vector<float> output(2 * n * batch);
    cudaMemcpy(output.data(), d_data, sizeof(float) * 2 * n * batch, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cufftDestroy(plan);

    return output;
}
