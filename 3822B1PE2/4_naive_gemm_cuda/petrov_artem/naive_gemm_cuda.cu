#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

constexpr int BLOCK_SIZE = 32;

__global__ void naive_gemm_simple(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n || col >= n) return;
    
    float sum = 0.0f;
    
    #pragma unroll 8
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    
    c[row * n + col] = sum;
}

__global__ void naive_gemm_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    
    const int TILE_SIZE = 4;
    
    int row = blockIdx.y * (blockDim.y * TILE_SIZE) + threadIdx.y * TILE_SIZE;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col >= n) return;
    
    for (int tile_row = 0; tile_row < TILE_SIZE; ++tile_row) {
        int cur_row = row + tile_row;
        if (cur_row >= n) break;
        
        float sum = 0.0f;
        const float* a_row = a + cur_row * n;
        
        #pragma unroll 8
        for (int k = 0; k < n; ++k) {
            sum += a_row[k] * b[k * n + col];
        }
        
        c[cur_row * n + col] = sum;
    }
}

__global__ void naive_gemm_kernel_shared(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    
    __shared__ float a_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_tile[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * BLOCK_SIZE + tx;
        if (row < n && a_col < n) {
            a_tile[ty][tx] = a[row * n + a_col];
        } else {
            a_tile[ty][tx] = 0.0f;
        }
        
        int b_row = t * BLOCK_SIZE + ty;
        if (b_row < n && col < n) {
            b_tile[ty][tx] = b[b_row * n + col];
        } else {
            b_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += a_tile[ty][k] * b_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n == 0) return {};
    
    if (a.size() != static_cast<size_t>(n * n) || 
        b.size() != static_cast<size_t>(n * n)) {
        std::cerr << "Error: Matrix size mismatch!" << std::endl;
        return {};
    }
    
    size_t bytes = n * n * sizeof(float);
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    naive_gemm_simple<<<grid, block>>>(d_a, d_b, d_c, n);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<float> c(n * n);
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    return c;
}
