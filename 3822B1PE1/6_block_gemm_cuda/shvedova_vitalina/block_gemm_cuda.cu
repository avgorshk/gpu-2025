#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cassert>

constexpr int TILE = 16;

__global__ void BlockMatMulKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    int numTiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aIdx = row * n + t * TILE + tx;
        int bIdx = (t * TILE + ty) * n + col;

        tileA[ty][tx] = (row < n && t * TILE + tx < n) ? A[aIdx] : 0.0f;
        tileB[ty][tx] = ((t * TILE + ty) < n && col < n) ? B[bIdx] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += tileA[ty][k] * tileB[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = acc;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    assert(a.size() == static_cast<size_t>(n * n));
    assert(b.size() == static_cast<size_t>(n * n));

    float *dA, *dB, *dC;
    size_t bytes = n * n * sizeof(float);

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    BlockMatMulKernel<<<grid, threads>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return c;
}
