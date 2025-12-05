#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

template<int BLOCK_SIZE>
__global__ void naiveGemmKernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int n) {
    const int tile_size = 4;
    const int row = blockIdx.y * blockDim.y * tile_size + threadIdx.y * tile_size;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum[tile_size] = {0.0f};
        
        for (int k = 0; k < n; k++) {
            float4 a_vec;
            const int a_base_idx = (row + 0) * n + k;
            a_vec.x = A[a_base_idx];
            a_vec.y = A[a_base_idx + n];
            a_vec.z = A[a_base_idx + 2 * n];
            a_vec.w = A[a_base_idx + 3 * n];
            
            float b_val = B[k * n + col];
            
            sum[0] += a_vec.x * b_val;
            sum[1] += a_vec.y * b_val;
            sum[2] += a_vec.z * b_val;
            sum[3] += a_vec.w * b_val;
        }
        
        if (row + 0 < n) C[(row + 0) * n + col] = sum[0];
        if (row + 1 < n) C[(row + 1) * n + col] = sum[1];
        if (row + 2 < n) C[(row + 2) * n + col] = sum[2];
        if (row + 3 < n) C[(row + 3) * n + col] = sum[3];
    }
}

template<int BLOCK_SIZE>
__global__ void naiveGemmSharedKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        if (row < n && (tile + threadIdx.x) < n) {
            As[threadIdx.y][threadIdx.x] = A[row * n + tile + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((tile + threadIdx.y) < n && col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile + threadIdx.y) * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n);
    
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);
    
    const int block_size = 16;
    dim3 blockDim(block_size, block_size);
    
    const int tile_size = 4;
    int grid_x = (n + block_size - 1) / block_size;
    int grid_y = (n + block_size * tile_size - 1) / (block_size * tile_size);
    dim3 gridDim(grid_x, grid_y);
    
    if (n <= 1024) {
        naiveGemmKernel<16><<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    } else {
        naiveGemmSharedKernel<16><<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return c;
}