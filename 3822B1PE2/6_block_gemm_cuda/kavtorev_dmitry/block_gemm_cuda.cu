#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void block_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_row_idx = row;
        int a_col_idx = tile * TILE_SIZE + threadIdx.x;
        int b_row_idx = tile * TILE_SIZE + threadIdx.y;
        int b_col_idx = col;
        
        if (a_row_idx < n && a_col_idx < n) {
            shared_a[threadIdx.y][threadIdx.x] = a[a_row_idx * n + a_col_idx];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (b_row_idx < n && b_col_idx < n) {
            shared_b[threadIdx.y][threadIdx.x] = b[b_row_idx * n + b_col_idx];
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
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
    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE,
                  (n + TILE_SIZE - 1) / TILE_SIZE);
    
    block_gemm_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}

