#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;

__global__ void naive_gemm_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    
    const int tile_size = BLOCK_SIZE;
    int I = blockIdx.y;
    int J = blockIdx.x;
    
    int i = I * tile_size + threadIdx.y;
    int j = J * tile_size + threadIdx.x;
    
    if (i >= n || j >= n) return;
    
    float sum = 0.0f;
    
    int block_count = (n + tile_size - 1) / tile_size;
    
    for (int K = 0; K < block_count; ++K) {
        for (int kk = 0; kk < tile_size; ++kk) {
            int k = K * tile_size + kk;
            if (k < n) {
                sum += a[i * n + k] * b[k * n + j];
            }
        }
    }
    
    c[i * n + j] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n == 0) return {};
    
    size_t bytes = n * n * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int grid_x = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_y = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_x, grid_y);
    
    naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}
