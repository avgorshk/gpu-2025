#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

const int TILE_SIZE = 32;

__global__ void block_gemm_kernel(const float* A, const float* B, float* C, int N) {
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float c_value = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int A_col = t * TILE_SIZE + tx;
        if (row < N && A_col < N) {
            As[ty][tx] = A[row * N + A_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        int B_row = t * TILE_SIZE + ty;
        if (B_row < N && col < N) {
            Bs[ty][tx] = B[B_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            c_value += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = c_value;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n == 0) return {};
    
    size_t size = n * n * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);
    
    int block_size = (n < 128) ? 16 : 32;
    if (n >= 512) block_size = 32;
    
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);
    
    block_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    
    cudaDeviceSynchronize();
    
    std::vector<float> result(n * n);
    cudaMemcpy(result.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return result;
}