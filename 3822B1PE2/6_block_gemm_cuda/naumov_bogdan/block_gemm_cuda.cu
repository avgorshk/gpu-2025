#include "block_gemm_cuda.h"
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE BLOCK_SIZE

template<int TILE_SZ>
__global__ void blockGemmKernel(const float* A, const float* B, float* C, 
                                int n) {
    __shared__ float As[TILE_SZ][TILE_SZ];
    __shared__ float Bs[TILE_SZ][TILE_SZ];
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    int row = blockRow * TILE_SZ + threadRow;
    int col = blockCol * TILE_SZ + threadCol;
    
    float sum = 0.0f;
    int numTiles = n / TILE_SZ;
    
    for (int t = 0; t < numTiles; ++t) {
        int aRow = blockRow * TILE_SZ + threadRow;
        int aCol = t * TILE_SZ + threadCol;
        As[threadRow][threadCol] = (aRow < n && aCol < n) ? 
                                   A[aRow * n + aCol] : 0.0f;
        
        int bRow = t * TILE_SZ + threadRow;
        int bCol = blockCol * TILE_SZ + threadCol;
        Bs[threadRow][threadCol] = (bRow < n && bCol < n) ? 
                                   B[bRow * n + bCol] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SZ; ++k) {
            sum += As[threadRow][k] * Bs[k][threadCol];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Invalid matrix size");
    }
    
    std::vector<float> c(n * n, 0.0f);
    float *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));
    
    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    blockGemmKernel<BLOCK_SIZE><<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}