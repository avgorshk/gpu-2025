#include "block_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

constexpr int BLOCK_DIM = 16;

__global__ void matrixMultiplyDoubleBuffered(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C,
                                             int n)
{
    __shared__ float tileA[2][BLOCK_DIM][BLOCK_DIM + 1];
    __shared__ float tileB[2][BLOCK_DIM][BLOCK_DIM + 1];

    const int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    const int col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    if (row >= n || col >= n)
        return;

    float value = 0.0f;
    const int numTiles = (n + BLOCK_DIM - 1) / BLOCK_DIM;

    int tiledColA = threadIdx.x;
    int tiledRowB = threadIdx.y;

    tileA[0][threadIdx.y][threadIdx.x] = (tiledColA < n) ? A[row * n + tiledColA] : 0.0f;
    tileB[0][threadIdx.y][threadIdx.x] = (tiledRowB < n) ? B[tiledRowB * n + col] : 0.0f;

    __syncthreads();

    for (int t = 0; t < numTiles; ++t)
    {
        const int current = t % 2;
        const int next = (t + 1) % 2;

        if (t + 1 < numTiles)
        {
            tiledColA = (t + 1) * BLOCK_DIM + threadIdx.x;
            tiledRowB = (t + 1) * BLOCK_DIM + threadIdx.y;

            tileA[next][threadIdx.y][threadIdx.x] = (tiledColA < n) ? A[row * n + tiledColA] : 0.0f;
            tileB[next][threadIdx.y][threadIdx.x] = (tiledRowB < n) ? B[tiledRowB * n + col] : 0.0f;
        }

#pragma unroll
        for (int k = 0; k < BLOCK_DIM; ++k)
        {
            value += tileA[current][threadIdx.y][k] * tileB[current][k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * n + col] = value;
}

__global__ void matrixMultiplyOptimized(const float *__restrict__ A,
                                        const float *__restrict__ B,
                                        float *__restrict__ C,
                                        int n)
{
    __shared__ float tileA[BLOCK_DIM][BLOCK_DIM + 1];
    __shared__ float tileB[BLOCK_DIM][BLOCK_DIM + 1];

    const int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    const int col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    if (row >= n || col >= n)
        return;

    float value = 0.0f;
    const int numTiles = (n + BLOCK_DIM - 1) / BLOCK_DIM;

    for (int t = 0; t < numTiles; ++t)
    {
        const int tiledColA = t * BLOCK_DIM + threadIdx.x;
        const int tiledRowB = t * BLOCK_DIM + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (tiledColA < n) ? A[row * n + tiledColA] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tiledRowB < n) ? B[tiledRowB * n + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_DIM; ++k)
        {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * n + col] = value;
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    if (n <= 0)
        return {};

    std::vector<float> c(n * n, 0.0f);
    const size_t size = n * n * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpyAsync(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, b.data(), size, cudaMemcpyHostToDevice);
    cudaMemsetAsync(d_C, 0, size);

    const dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    const dim3 gridSize((n + BLOCK_DIM - 1) / BLOCK_DIM,
                        (n + BLOCK_DIM - 1) / BLOCK_DIM);

    if (n >= 512)
    {
        matrixMultiplyDoubleBuffered<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    }
    else
    {
        matrixMultiplyOptimized<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);

    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    return c;
}