#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void naive_gemm_optimized_kernel(const float* __restrict__ a, 
                                             const float* __restrict__ b, 
                                             float* __restrict__ c, 
                                             int n) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; ++tile) {
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;
        
        if (row < n && a_col < n) {
            tile_a[ty][tx] = a[row * n + a_col];
        } else {
            tile_a[ty][tx] = 0.0f;
        }
        
        if (b_row < n && col < n) {
            tile_b[ty][tx] = b[b_row * n + col];
        } else {
            tile_b[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        if (row < n && col < n) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += tile_a[ty][k] * tile_b[k][tx];
            }
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n);
    
    float *d_a, *d_b, *d_c;
    size_t size = n * n * sizeof(float);
    
    cudaError_t err;
    
    err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_a: " << cudaGetErrorString(err) << std::endl;
        return c;
    }
    
    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_b: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a);
        return c;
    }
    
    err = cudaMalloc((void**)&d_c, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_c: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        return c;
    }
    
    err = cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_a: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return c;
    }
    
    err = cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_b: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return c;
    }
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    
    naive_gemm_optimized_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for result: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}

