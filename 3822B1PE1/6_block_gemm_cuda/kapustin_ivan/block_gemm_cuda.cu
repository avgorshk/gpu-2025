#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

static constexpr int TILE = 16;

__global__ void tiledGemmKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int globalRow = blockIdx.y * TILE + ty;
    const int globalCol = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    const int tilesCount = (n + TILE - 1) / TILE;

    for (int t = 0; t < tilesCount; ++t) {
        const int aIndex = globalRow * n + (t * TILE + tx);
        const int bIndex = (t * TILE + ty) * n + globalCol;

        tileA[ty][tx] = (globalRow < n && (t * TILE + tx) < n) ? A[aIndex] : 0.0f;
        tileB[ty][tx] = ((t * TILE + ty) < n && globalCol < n) ? B[bIndex] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (globalRow < n && globalCol < n) {
        C[globalRow * n + globalCol] = sum;
    }
}

__global__ void tiledGemmKernelOptimized(const float* A, const float* B, float* C, int n) {
    __shared__ float bufA[TILE][TILE];
    __shared__ float bufB[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float result = 0.0f;
    const int segments = (n + TILE - 1) / TILE;

    for (int seg = 0; seg < segments; ++seg) {
        const int aR = row;
        const int aC = seg * TILE + tx;
        bufA[ty][tx] = (aR < n && aC < n) ? A[aR * n + aC] : 0.0f;

        const int bR = seg * TILE + ty;
        const int bC = col;
        bufB[ty][tx] = (bR < n && bC < n) ? B[bR * n + bC] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            result += bufA[ty][k] * bufB[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = result;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {

    assert(a.size() == static_cast<size_t>(n * n));
    assert(b.size() == static_cast<size_t>(n * n));
    assert((n & (n - 1)) == 0 && "Matrix dimension must be power of two.");

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t bytes = sizeof(float) * n * n;

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    tiledGemmKernelOptimized<<<blocks, threads>>>(dA, dB, dC, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: "
                  << cudaGetErrorString(err)
                  << std::endl;

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return {};
    }

    std::vector<float> result(n * n);
    cudaMemcpy(result.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return result;
}