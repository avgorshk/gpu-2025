#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void BlockGemmKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    int numBlocks = n / BLOCK_SIZE;
    
    for (int m = 0; m < numBlocks; m++) {
        int aCol = m * BLOCK_SIZE + threadIdx.x;
        int aIndex = row * n + aCol;
        if (row < n && aCol < n) {
            As[threadIdx.y][threadIdx.x] = A[aIndex];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int bRow = m * BLOCK_SIZE + threadIdx.y;
        int bIndex = bRow * n + col;
        if (bRow < n && col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[bIndex];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
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
    std::vector<float> c(n * n);
    
    float *d_a, *d_b, *d_c;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    BlockGemmKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}