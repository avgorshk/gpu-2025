#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <vector>

constexpr int BLOCK_SIZE = 16;

__global__ void transpose_kernel(const float* __restrict__ input,
    float* __restrict__ output,
    int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

__global__ void naive_gemm_transposed_kernel(const float* __restrict__ a,
    const float* __restrict__ b_t,
    float* __restrict__ c,
    int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;

        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b_t[col * n + k];
        }

        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Matrix sizes do not match n*n");
    }

    size_t bytes = n * n * sizeof(float);
    std::vector<float> c(n * n);

    if (n == 0) return c;

    float* d_a = nullptr, * d_b = nullptr, * d_b_t = nullptr, * d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_b_t, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 blockSizeTranspose(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSizeTranspose((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    transpose_kernel << <gridSizeTranspose, blockSizeTranspose >> > (d_b, d_b_t, n);
    cudaDeviceSynchronize();

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    naive_gemm_transposed_kernel << <gridSize, blockSize >> > (d_a, d_b_t, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_b_t);
    cudaFree(d_c);

    return c;
}