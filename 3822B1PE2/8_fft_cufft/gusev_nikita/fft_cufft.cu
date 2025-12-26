#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult result = call; \
        if (result != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT error: " << result << std::endl; \
        } \
    } while(0)

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
    CUDA_CHECK(cudaMalloc(&d_data, total_size * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemcpy(d_data, input.data(), total_size * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_C2C, batch));
    
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    normalize<<<num_blocks, block_size>>>(d_data, n, total_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(output.data(), d_data, total_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_data));
    
    return output;
}

