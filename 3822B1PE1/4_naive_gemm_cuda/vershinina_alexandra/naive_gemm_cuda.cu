#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdint>

inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }
}

#ifndef TILE
#define TILE 16
#endif

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float acc = 0.0f;

    for (int m = 0; m < N; m += TILE) {
        if (row < N && (m + tx) < N)
            sA[ty][tx] = A[row * N + (m + tx)];
        else
            sA[ty][tx] = 0.0f;

        if ((m + ty) < N && col < N)
            sB[ty][tx] = B[(m + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};
    const size_t N = static_cast<size_t>(n);
    const size_t elems = N * N;
    if (a.size() != elems || b.size() != elems) {
        throw std::invalid_argument("Input matrices must have size n*n");
    }

    const size_t bytes = elems * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    try {
        checkCuda(cudaMalloc((void**)&dA, bytes), "cudaMalloc dA");
        checkCuda(cudaMalloc((void**)&dB, bytes), "cudaMalloc dB");
        checkCuda(cudaMalloc((void**)&dC, bytes), "cudaMalloc dC");

        checkCuda(cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D dA");
        checkCuda(cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D dB");

        dim3 block(TILE, TILE);
        dim3 grid( static_cast<unsigned int>((N + TILE - 1) / TILE),
                   static_cast<unsigned int>((N + TILE - 1) / TILE) );

        gemm_tiled_kernel<<<grid, block>>>(dA, dB, dC, n);
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "kernel execution");

        std::vector<float> C(elems);
        checkCuda(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H dC");

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        return C;
    } catch (...) {
        if (dA) cudaFree(dA);
        if (dB) cudaFree(dB);
        if (dC) cudaFree(dC);
        throw;
    }
}
