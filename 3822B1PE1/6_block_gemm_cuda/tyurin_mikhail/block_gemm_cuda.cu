#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int block = 0; block < n; block += BLOCK_SIZE) {
        if (row < n && (block + threadIdx.x) < n)
            sA[threadIdx.y][threadIdx.x] = A[row * n + block + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((block + threadIdx.y) < n && col < n)
            sB[threadIdx.y][threadIdx.x] = B[(block + threadIdx.y) * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& A,
                                 const std::vector<float>& B,
                                 int n) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    float *dA, *dB, *dC;

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    block_gemm_kernel<<<grid, block>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    std::vector<float> C(n * n);
    cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return C;
}
