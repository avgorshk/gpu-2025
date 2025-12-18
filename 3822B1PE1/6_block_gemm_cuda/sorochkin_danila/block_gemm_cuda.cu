#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 16;

__global__ void block_gemm_kernel(const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n) {
    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;
    int block_count = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; 

    for (int bk = 0; bk < block_count; ++bk) {
        if (row < n && (bk * BLOCK_SIZE + threadIdx.x) < n) {
            a_shared[threadIdx.y][threadIdx.x] = a[row * n + bk * BLOCK_SIZE + threadIdx.x];
        }
        else {
            a_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row_idx = bk * BLOCK_SIZE + threadIdx.y;
        if (b_row_idx < n && col < n) {
            b_shared[threadIdx.y][threadIdx.x] = b[b_row_idx * n + col];
        }
        else {
            b_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
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
    if (n == 0) return {};

    int grid_x = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_y = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t bytes = n * n * sizeof(float);
    float* d_a = nullptr, * d_b = nullptr, * d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(grid_x, grid_y);

    block_gemm_kernel << <grid_dim, block_dim >> > (d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}