#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 32

__global__ void gemm_tile_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row_idx = blockIdx.y * TILE_WIDTH + ty;
    int col_idx = blockIdx.x * TILE_WIDTH + tx;

    float acc = 0.0f;

    int num_tiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < num_tiles; ++t) {
        int load_col_a = t * TILE_WIDTH + tx;
        int load_row_b = t * TILE_WIDTH + ty;

        if (row_idx < n && load_col_a < n) {
            tile_a[ty][tx] = a[row_idx * n + load_col_a];
        } else {
            tile_a[ty][tx] = 0.0f;
        }

        if (col_idx < n && load_row_b < n) {
            tile_b[ty][tx] = b[load_row_b * n + col_idx];
        } else {
            tile_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += tile_a[ty][k] * tile_b[k][tx];
        }

        __syncthreads();
    }

    if (row_idx < n && col_idx < n) {
        c[row_idx * n + col_idx] = acc;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> c(n * n);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks_per_grid((n + TILE_WIDTH - 1) / TILE_WIDTH,
                         (n + TILE_WIDTH - 1) / TILE_WIDTH);

    gemm_tile_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_a, d_b, d_c, n);

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}
