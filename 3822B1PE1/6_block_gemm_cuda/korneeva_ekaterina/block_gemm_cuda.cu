#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <iostream>

#define TILE_SIZE 32

__global__ void blockGemmOptimizedKernel(const float* __restrict__ a, 
                                         const float* __restrict__ b, 
                                         float* __restrict__ c, 
                                         int n) {
    
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE + 1];
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    int row = blockRow * TILE_SIZE + threadRow;
    int col = blockCol * TILE_SIZE + threadCol;
    
    float sum = 0.0f;
    
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        int aRow = row;
        int aCol = t * TILE_SIZE + threadCol;
        float aValue = (aRow < n && aCol < n) ? a[aRow * n + aCol] : 0.0f;
        
        int bRow = t * TILE_SIZE + threadRow;
        int bCol = col;
        float bValue = (bRow < n && bCol < n) ? b[bRow * n + bCol] : 0.0f;
        
        sharedA[threadRow][threadCol] = aValue;
        sharedB[threadRow][threadCol] = bValue;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[threadRow][k] * sharedB[k][threadCol];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    
    assert(a.size() == static_cast<size_t>(n * n));
    assert(b.size() == static_cast<size_t>(n * n));
    
    std::vector<float> c(n * n, 0.0f);
    
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    size_t matrixSize = n * n * sizeof(float);
    cudaMalloc(&d_a, matrixSize);
    cudaMalloc(&d_b, matrixSize);
    cudaMalloc(&d_c, matrixSize);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMemcpyAsync(d_a, a.data(), matrixSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), matrixSize, cudaMemcpyHostToDevice, stream);
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, 
                 (n + TILE_SIZE - 1) / TILE_SIZE);
    
    int sharedMemSize = 2 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(float);
    blockGemmOptimizedKernel<<<gridDim, blockDim, sharedMemSize, stream>>>(d_a, d_b, d_c, n);
    
    cudaMemcpyAsync(c.data(), d_c, matrixSize, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}