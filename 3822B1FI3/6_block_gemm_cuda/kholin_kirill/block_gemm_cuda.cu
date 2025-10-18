#include "block_gemm_cuda.h"
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE BLOCK_SIZE

__global__ void block_gemm_kernel(const float* __restrict__ dev_a, 
                                  const float* __restrict__ dev_b, 
                                  float* __restrict__ dev_c, 
                                  int matrix_size) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= matrix_size || col >= matrix_size) {
        return;
    }

    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int num_tiles = (matrix_size + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        int a_col = tile_idx * TILE_SIZE + threadIdx.x;
        if (row < matrix_size && a_col < matrix_size) {
            tile_a[threadIdx.y][threadIdx.x] = dev_a[row * matrix_size + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = tile_idx * TILE_SIZE + threadIdx.y;
        if (b_row < matrix_size && col < matrix_size) {
            tile_b[threadIdx.y][threadIdx.x] = dev_b[b_row * matrix_size + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < matrix_size && col < matrix_size) {
        dev_c[row * matrix_size + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t matrix_bytes = n * n * sizeof(float);
    
    float* dev_a;
    float* dev_b; 
    float* dev_c;
    
    cudaMalloc(&dev_a, matrix_bytes);
    cudaMalloc(&dev_b, matrix_bytes);
    cudaMalloc(&dev_c, matrix_bytes);

    cudaMemcpy(dev_a, a.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), matrix_bytes, cudaMemcpyHostToDevice);
    
    cudaMemset(dev_c, 0, matrix_bytes);

    dim3 block_dims(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dims((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    block_gemm_kernel<<<grid_dims, block_dims>>>(dev_a, dev_b, dev_c, n);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(cuda_err) << std::endl;
    }

    cudaDeviceSynchronize();

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), dev_c, matrix_bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return c;
}