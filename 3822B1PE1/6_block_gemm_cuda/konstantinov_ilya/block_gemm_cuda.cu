#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16 

__global__ void blockGemmKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < n / BLOCK_SIZE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * n + t * BLOCK_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * n + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n)
        C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = n * n * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(n / BLOCK_SIZE, n / BLOCK_SIZE);
    blockGemmKernel<<<blocks, threads>>>(dA, dB, dC, n);
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return c;
}