#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void BlockGemmKernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= n || col >= n) return;

    __shared__ float s_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE * BLOCK_SIZE];

    float result = 0.0f;

    for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        int a_idx = row * n + (k * BLOCK_SIZE + threadIdx.x);
        s_A[threadIdx.y * BLOCK_SIZE + threadIdx.x] =
            (row < n && (k * BLOCK_SIZE + threadIdx.x) < n) ? A[a_idx] : 0.0f;

        int b_idx = (k * BLOCK_SIZE + threadIdx.y) * n + col;
        s_B[threadIdx.y * BLOCK_SIZE + threadIdx.x] =
            (col < n && (k * BLOCK_SIZE + threadIdx.y) < n) ? B[b_idx] : 0.0f;

        __syncthreads();

        for (int t = 0; t < BLOCK_SIZE; ++t) {
            result += s_A[threadIdx.y * BLOCK_SIZE + t] * s_B[t * BLOCK_SIZE + threadIdx.x];
        }

        __syncthreads();
    }
    C[row * n + col] = result;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                          const std::vector<float>& b,
                                          int n) {

    size_t memory = n * n * sizeof(float);
    float* d_A, * d_B, * d_C;
    std::vector<float> result(n * n);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaMalloc(&d_A, memory);
    cudaMalloc(&d_B, memory);
    cudaMalloc(&d_C, memory);

    cudaMemcpy(d_A, a.data(), memory, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), memory, cudaMemcpyHostToDevice);

    BlockGemmKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, n);
    cudaMemcpy(result.data(), d_C, memory, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}