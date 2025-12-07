#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void blockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float shared_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int tile = 0; tile < num_tiles; ++tile) {
        int a_col = tile * BLOCK_SIZE + threadIdx.x;
        if (row < n && a_col < n) {
            shared_a[threadIdx.y][threadIdx.x] = a[row * n + a_col];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int b_row = tile * BLOCK_SIZE + threadIdx.y;
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
    std::vector<float> c(n * n);
    
    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static int allocated_size = 0;
    
    int size = n * n * sizeof(float);
    
    if (allocated_size < size) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
        allocated_size = size;
    }
    
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    blockGemmKernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    return c;
}
