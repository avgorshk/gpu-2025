#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

constexpr int BLOCK_SIZE = 16;
__constant__ int tiles;

__global__ void block_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, 
                                   float* __restrict__ C, int n) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE + 1];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < tiles; ++t) {
        int tileCol = t * BLOCK_SIZE + threadIdx.x;
        int tileRow = t * BLOCK_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < n && tileCol < n) ? A[row * n + tileCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (tileRow < n && col < n) ? B[tileRow * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c(n * n);
    const size_t bytes = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    const dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int ltiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMemcpyToSymbol(tiles, &ltiles, sizeof(int));

    block_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}
