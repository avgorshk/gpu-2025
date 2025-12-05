#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUFFT_ERROR(call) \
    do { \
        cufftResult result = call; \
        if (result != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << result << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void normalizeKernel(cufftComplex* data, int n, int total_elements, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    
    cufftHandle plan_forward, plan_inverse;
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch));
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch));
    
    cufftComplex* d_data;
    size_t size = n * batch * sizeof(cufftComplex);
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, input.data(), size, cudaMemcpyHostToDevice));
    
    CHECK_CUFFT_ERROR(cufftExecC2C(plan_forward, d_data, d_data, CUFFT_FORWARD));
    
    CHECK_CUFFT_ERROR(cufftExecC2C(plan_inverse, d_data, d_data, CUFFT_INVERSE));
    
    float scale = 1.0f / n;
    int total_elements = n * batch;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    normalizeKernel<<<numBlocks, blockSize>>>(d_data, n, total_elements, scale);
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    std::vector<float> output(input.size());
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_data, size, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUFFT_ERROR(cufftDestroy(plan_forward));
    CHECK_CUFFT_ERROR(cufftDestroy(plan_inverse));
    
    return output;
}