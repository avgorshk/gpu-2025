#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

#define TILE_SIZE 32
#define UNROLL_FACTOR 4

__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n || col >= n) return;
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        sum += A[row * n + k] * B[k * n + col];
    }
    
    C[row * n + col] = sum;
}

__global__ void optimized_gemm_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int n) {
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    

    float sum[TILE_SIZE/TILE_SIZE] = {0.0f};
    
    for (int k = 0; k < n; k += UNROLL_FACTOR) {
        // Предзагрузка нескольких значений
        float a_vals[UNROLL_FACTOR];
        float b_vals[UNROLL_FACTOR];
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR && (k + u) < n; u++) {
            a_vals[u] = A[row * n + (k + u)];
            b_vals[u] = B[(k + u) * n + col];
        }
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR && (k + u) < n; u++) {
            sum[0] += a_vals[u] * b_vals[u];
        }
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum[0];
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
   
    assert(a.size() == static_cast<size_t>(n * n));
    assert(b.size() == static_cast<size_t>(n * n));
    
    std::vector<float> c(n * n, 0.0f);
    
    float *d_a, *d_b, *d_c;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_a, a.data(), size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), size, cudaMemcpyHostToDevice, stream);
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, 
                 (n + TILE_SIZE - 1) / TILE_SIZE);
    
    naive_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, n);
   
    cudaMemcpyAsync(c.data(), d_c, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}