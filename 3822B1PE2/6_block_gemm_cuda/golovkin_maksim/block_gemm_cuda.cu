#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        int bRow = t * BLOCK_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? a[row * n + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? b[bRow * n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
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
    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    BlockGemmKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
