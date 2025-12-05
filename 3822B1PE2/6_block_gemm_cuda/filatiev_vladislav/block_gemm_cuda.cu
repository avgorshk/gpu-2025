#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>

#define TILE_SIZE 32

__global__ void block_gemm_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int k = 0; k < N; k += TILE_SIZE) {
        if (row < N && k + tx < N)
            As[ty][tx] = A[row * N + k + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && k + ty < N)
            Bs[ty][tx] = B[(k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    block_gemm_kernel << <grid, block >> > (d_a, d_b, d_c, n);

    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}