#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <stdexcept>
#include <cassert>

__global__ void normalize_kernel(cufftComplex* data, float scale, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (batch <= 0) {
        throw std::invalid_argument("batch must be positive");
    }
    if (input.empty() || input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("input size must be divisible by 2*batch");
    }
    
    const size_t n = input.size() / (2 * batch);

    static cufftHandle plan_forward = 0;
    static cufftHandle plan_inverse = 0;
    static size_t cached_n = 0;
    static int cached_batch = 0;
    static cufftComplex* d_data = nullptr;
    static size_t cached_data_size = 0;

    bool recreate_plan = (plan_forward == 0) || (n != cached_n) || (batch != cached_batch);
    bool recreate_memory = (d_data == nullptr) || (input.size() * sizeof(float) != cached_data_size);
    
    if (recreate_plan) {
        if (plan_forward != 0) cufftDestroy(plan_forward);
        if (plan_inverse != 0) cufftDestroy(plan_inverse);
        
        cufftResult_t cufftStatus = cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch);
        if (cufftStatus != CUFFT_SUCCESS) {
            throw std::runtime_error("cufftPlan1d failed for forward plan");
        }
        
        cufftStatus = cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch);
        if (cufftStatus != CUFFT_SUCCESS) {
            cufftDestroy(plan_forward);
            throw std::runtime_error("cufftPlan1d failed for inverse plan");
        }
        
        cached_n = n;
        cached_batch = batch;
    }
    
    if (recreate_memory) {
        if (d_data != nullptr) cudaFree(d_data);
        
        size_t dataSize = input.size() * sizeof(float);
        cudaError_t cudaStatus = cudaMalloc(&d_data, dataSize);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for d_data");
        }
        
        cached_data_size = dataSize;
    }
    
    cudaError_t cudaStatus = cudaMemcpy(d_data, input.data(), input.size() * sizeof(float), 
                                        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed for input data");
    }
    
    cufftResult_t cufftStatus = cufftExecC2C(plan_forward, d_data, d_data, CUFFT_FORWARD);
    if (cufftStatus != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftExecC2C failed for forward transform");
    }
    
    cufftStatus = cufftExecC2C(plan_inverse, d_data, d_data, CUFFT_INVERSE);
    if (cufftStatus != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftExecC2C failed for inverse transform");
    }
    
    const float scale = 1.0f / static_cast<float>(n);
    const size_t totalComplexNumbers = n * batch;
    
    const int blockSize = 256;
    const int gridSize = (totalComplexNumbers + blockSize - 1) / blockSize;
    
    normalize_kernel<<<gridSize, blockSize>>>(d_data, scale, totalComplexNumbers);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed for normalization");
    }
    
    std::vector<float> result(input.size());
    
    cudaStatus = cudaMemcpy(result.data(), d_data, input.size() * sizeof(float), 
                           cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed for result data");
    }

    return result;
}

void cleanup_cufft() {
    static cufftHandle plan_forward = 0;
    static cufftHandle plan_inverse = 0;
    static cufftComplex* d_data = nullptr;
    
    if (plan_forward != 0) cufftDestroy(plan_forward);
    if (plan_inverse != 0) cufftDestroy(plan_inverse);
    if (d_data != nullptr) cudaFree(d_data);
    
    plan_forward = 0;
    plan_inverse = 0;
    d_data = nullptr;
}