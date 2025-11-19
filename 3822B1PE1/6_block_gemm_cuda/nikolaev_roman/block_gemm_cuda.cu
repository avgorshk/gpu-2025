#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

constexpr int BLOCK_SIZE = 16;

__global__ void blockGemmKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    int row = blockRow * BLOCK_SIZE + threadY;
    int col = blockCol * BLOCK_SIZE + threadX;
    
    float cValue = 0.0f;
    
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int block = 0; block < numBlocks; ++block) {
        int aRow = blockRow * BLOCK_SIZE + threadY;
        int aCol = block * BLOCK_SIZE + threadX;
        if (aRow < n && aCol < n) {
            As[threadY][threadX] = A[aRow * n + aCol];
        } else {
            As[threadY][threadX] = 0.0f;
        }
        
        int bRow = block * BLOCK_SIZE + threadY;
        int bCol = blockCol * BLOCK_SIZE + threadX;
        if (bRow < n && bCol < n) {
            Bs[threadY][threadX] = B[bRow * n + bCol];
        } else {
            Bs[threadY][threadX] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            cValue += As[threadY][k] * Bs[k][threadX];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = cValue;
    }
}

__global__ void blockGemmOptimizedKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    int row = blockRow * BLOCK_SIZE + threadY;
    int col = blockCol * BLOCK_SIZE + threadX;
    
    float cValue = 0.0f;
    
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int block = 0; block < numBlocks; ++block) {
        int aRow = blockRow * BLOCK_SIZE + threadY;
        int aCol = block * BLOCK_SIZE + threadX;
        As[threadY][threadX] = (aRow < n && aCol < n) ? A[aRow * n + aCol] : 0.0f;
        
        int bRow = block * BLOCK_SIZE + threadY;
        int bCol = blockCol * BLOCK_SIZE + threadX;
        Bs[threadY][threadX] = (bRow < n && bCol < n) ? B[bRow * n + bCol] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            cValue += As[threadY][k] * Bs[k][threadX];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = cValue;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    assert(a.size() == n * n && b.size() == n * n);
    assert((n & (n - 1)) == 0 && "Matrix size must be power of 2");
    
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    blockGemmOptimizedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return std::vector<float>();
    }
    
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return c;
}