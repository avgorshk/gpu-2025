#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 32
__constant__ int c_numBlocks;

__global__ void blockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockIdx.y * BLOCK_SIZE + threadRow;
    int col = blockIdx.x * BLOCK_SIZE + threadCol;

    float sum = 0.0f;

    for (int k = 0; k < c_numBlocks; ++k) {
        int aRow = blockIdx.y * BLOCK_SIZE + threadRow;
        int aCol = k * BLOCK_SIZE + threadCol;
        As[threadRow][threadCol] = (aRow < n && aCol < n) ? a[aRow * n + aCol] : 0.0f;

        int bRow = k * BLOCK_SIZE + threadRow;
        int bCol = blockIdx.x * BLOCK_SIZE + threadCol;
        Bs[threadRow][threadCol] = (bRow < n && bCol < n) ? b[bRow * n + bCol] : 0.0f;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i += 4) {
            sum += As[threadRow][i] * Bs[i][threadCol];
            sum += As[threadRow][i + 1] * Bs[i + 1][threadCol];
            sum += As[threadRow][i + 2] * Bs[i + 2][threadCol];
            sum += As[threadRow][i + 3] * Bs[i + 3][threadCol];
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
    size_t matrixSize = n * n;
    size_t bytes = matrixSize * sizeof(float);
    std::vector<float> c(matrixSize, 0.0f);

    float *d_a{}, *d_b{}, *d_c{};

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMemcpyToSymbol(c_numBlocks, &numBlocks, sizeof(int));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(numBlocks, numBlocks);

    blockGemmKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}