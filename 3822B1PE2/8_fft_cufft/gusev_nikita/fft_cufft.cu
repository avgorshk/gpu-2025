#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void normalize(cufftComplex* data, int n, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    int total_size = n * batch;
    
    std::vector<float> output(input.size());
    
    cufftComplex* d_data;
    cudaMalloc(&d_data, total_size * sizeof(cufftComplex));
    cudaMemcpy(d_data, input.data(), total_size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    normalize<<<num_blocks, block_size>>>(d_data, n, total_size);
    
    cudaMemcpy(output.data(), d_data, total_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(d_data);
    
    return output;
}

