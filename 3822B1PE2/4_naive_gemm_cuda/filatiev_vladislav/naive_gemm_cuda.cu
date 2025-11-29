#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <vector>

template<int BLOCK_SIZE>
__global__ void gemm_fast(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int k = 0; k < N; k += BLOCK_SIZE) {
        if (row < N && k + tx < N)
            As[ty][tx] = A[row * N + k + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && k + ty < N)
            Bs[ty][tx] = B[(k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("invalid size");
    }

    std::vector<float> c(n * n);
    if (n == 0) return c;

    float* d_a, * d_b, * d_c;
    size_t size = n * n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    if (block_size == 16) {
        gemm_fast<16> << <grid, block >> > (d_a, d_b, d_c, n);
    }
    else if (block_size == 32) {
        gemm_fast<32> << <grid, block >> > (d_a, d_b, d_c, n);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}