#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

constexpr int BLOCK_SIZE = 32;

__global__ void blockGemmKernel(const float* A,
                                const float* B,
                                float* C,
                                int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float acc = 0.0f;

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        int a_col = tile + threadIdx.x;
        int b_row = tile + threadIdx.y;

        if (row < n && a_col < n) {
            As[threadIdx.y][threadIdx.x] = A[row * n + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && b_row < n) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    blockGemmKernel<<<grid, block>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    std::vector<float> c(static_cast<size_t>(n) * n);
    cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return c;
}
