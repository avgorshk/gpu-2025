#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void blockGemmKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    int row = blockRow * BLOCK_SIZE + ty;
    int col = blockCol * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int m = 0; m < n / BLOCK_SIZE; ++m) {
        int aRow = blockRow * BLOCK_SIZE + ty;
        int aCol = m * BLOCK_SIZE + tx;
        int bRow = m * BLOCK_SIZE + ty;
        int bCol = blockCol * BLOCK_SIZE + tx;
        
        if (aRow < n && aCol < n) {
            As[ty][tx] = A[aRow * n + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (bRow < n && bCol < n) {
            Bs[ty][tx] = B[bRow * n + bCol];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Input matrices must have size n*n");
    }
    
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    blockGemmKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    std::vector<float> c(n * n);
    CHECK_CUDA_ERROR(cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    
    return c;
}