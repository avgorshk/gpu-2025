#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>

#define BLOCK_SIZE 16
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

__global__ void blockGemmKernelSafe(const float* a, const float* b, float* c, int n) {
    __shared__ float shared_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    int row = block_row * BLOCK_SIZE + ty;
    int col = block_col * BLOCK_SIZE + tx;
    
    float c_value = 0.0f;
    
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int block = 0; block < num_blocks; ++block) {
        int a_col = block * BLOCK_SIZE + tx;
        if (row < n && a_col < n) {
            shared_a[ty][tx] = a[row * n + a_col];
        } else {
            shared_a[ty][tx] = 0.0f;
        }
        
        int b_row = block * BLOCK_SIZE + ty;
        if (b_row < n && col < n) {
            shared_b[ty][tx] = b[b_row * n + col];
        } else {
            shared_b[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            c_value += shared_a[ty][k] * shared_b[k][tx];
        }
        
        __syncthreads();
    }
    if (row < n && col < n) {
        c[row * n + col] = c_value;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);
    
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    
    try {
        CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
        CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));
        
        CHECK_CUDA_ERROR(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice));
        
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                     (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        blockGemmKernelSafe<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
        
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost));
        
    } catch (...) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        throw;
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}