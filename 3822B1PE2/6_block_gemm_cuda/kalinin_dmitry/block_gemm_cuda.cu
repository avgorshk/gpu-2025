#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void block_gemm_kernel_optimized(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            float* __restrict__ c,
                                            int n) {
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE + 1];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        int a_row = row;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        
        shared_a[threadIdx.y][threadIdx.x] = (a_row < n && a_col < n) ? a[a_row * n + a_col] : 0.0f;
        shared_b[threadIdx.y][threadIdx.x] = (b_row < n && b_col < n) ? b[b_row * n + b_col] : 0.0f;
        
        __syncthreads();
        
        int k = 0;
        for (; k < TILE_SIZE - 3; k += 4) {
            sum += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
            sum += shared_a[threadIdx.y][k + 1] * shared_b[k + 1][threadIdx.x];
            sum += shared_a[threadIdx.y][k + 2] * shared_b[k + 2][threadIdx.x];
            sum += shared_a[threadIdx.y][k + 3] * shared_b[k + 3][threadIdx.x];
        }
        for (; k < TILE_SIZE; ++k) {
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
    
    block_gemm_kernel_optimized<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}

