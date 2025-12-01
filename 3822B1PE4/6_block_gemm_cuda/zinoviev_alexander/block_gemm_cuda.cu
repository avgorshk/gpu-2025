#include "block_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void block_gemm_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float shared_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int a_col = block_idx * BLOCK_SIZE + threadIdx.x;
        int b_row = block_idx * BLOCK_SIZE + threadIdx.y;
        
        if (row < n && a_col < n) {
            shared_a[threadIdx.y][threadIdx.x] = a[row * n + a_col];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (b_row < n && col < n) {
            shared_b[threadIdx.y][threadIdx.x] = b[b_row * n + col];
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
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
    const size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);
    
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    block_gemm_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}