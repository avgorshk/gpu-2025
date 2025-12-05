#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>

#define BLOCK_DIM 32

__global__ void gemm_block_shared(const float* A, const float* B, float* C, int n, int tile_sz) {
    __shared__ float tile_a[BLOCK_DIM][BLOCK_DIM];
    __shared__ float tile_b[BLOCK_DIM][BLOCK_DIM];

    int global_row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int global_col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    float accum = 0.0f;
    int num_blocks = (n + tile_sz - 1) / tile_sz;

    for (int blk = 0; blk < num_blocks; ++blk) {
        int load_col = blk * tile_sz + threadIdx.x;
        int load_row = blk * tile_sz + threadIdx.y;

        if (global_row < n && load_col < n) {
            tile_a[threadIdx.y][threadIdx.x] = A[global_row * n + load_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (load_row < n && global_col < n) {
            tile_b[threadIdx.y][threadIdx.x] = B[load_row * n + global_col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int idx = 0; idx < BLOCK_DIM; ++idx) {
            accum += tile_a[threadIdx.y][idx] * tile_b[idx][threadIdx.x];
        }

        __syncthreads();
    }

    if (global_row < n && global_col < n) {
        C[global_row * n + global_col] = accum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (a.size() != static_cast<size_t>(n * n) || 
        b.size() != static_cast<size_t>(n * n) || 
        n <= 0) {
        return {};
    }

    size_t size_bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> result(static_cast<size_t>(n) * n, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaStream_t str;
    if (cudaStreamCreate(&str) != cudaSuccess) {
        return result;
    }

    if (cudaMalloc(&d_a, size_bytes) != cudaSuccess ||
        cudaMalloc(&d_b, size_bytes) != cudaSuccess ||
        cudaMalloc(&d_c, size_bytes) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaStreamDestroy(str);
        return result;
    }

    if (cudaMemcpyAsync(d_a, a.data(), size_bytes, cudaMemcpyHostToDevice, str) != cudaSuccess ||
        cudaMemcpyAsync(d_b, b.data(), size_bytes, cudaMemcpyHostToDevice, str) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaStreamDestroy(str);
        return result;
    }

    int grid_x = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    int grid_y = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 grid_dim(grid_x, grid_y);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);

    gemm_block_shared<<<grid_dim, block_dim, 0, str>>>(d_a, d_b, d_c, n, BLOCK_DIM);

    if (cudaMemcpyAsync(result.data(), d_c, size_bytes, cudaMemcpyDeviceToHost, str) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaStreamDestroy(str);
        return result;
    }

    cudaStreamSynchronize(str);
    cudaStreamDestroy(str);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}
