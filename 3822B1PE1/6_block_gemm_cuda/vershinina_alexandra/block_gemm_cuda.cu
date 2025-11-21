#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <string>

#ifndef TILE
#define TILE 16
#endif

#ifndef PAD
#define PAD 1
#endif

inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }
}

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    __shared__ float As[TILE][TILE + PAD];
    __shared__ float Bs[TILE][TILE + PAD];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y; 

    const int row = by * TILE + ty;
    const int col = bx * TILE + tx;

    float acc = 0.0f;

    const int numTiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        if (row < n && a_col < n) {
            As[ty][tx] = A[row * n + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (b_row < n && col < n) {
            Bs[ty][tx] = B[b_row * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
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
    if (n <= 0) return {};
    const size_t N = static_cast<size_t>(n);
    const size_t elems = N * N;
    if (a.size() != elems || b.size() != elems) {
        throw std::invalid_argument("Input matrices must be size n*n");
    }

    const size_t bytes = elems * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    float *pinnedA = nullptr, *pinnedB = nullptr, *pinnedC = nullptr;
    cudaStream_t stream = nullptr;

    try {
        checkCuda(cudaMalloc((void**)&dA, bytes), "cudaMalloc dA");
        checkCuda(cudaMalloc((void**)&dB, bytes), "cudaMalloc dB");
        checkCuda(cudaMalloc((void**)&dC, bytes), "cudaMalloc dC");

        checkCuda(cudaMallocHost((void**)&pinnedA, bytes), "cudaMallocHost pinnedA");
        checkCuda(cudaMallocHost((void**)&pinnedB, bytes), "cudaMallocHost pinnedB");
        checkCuda(cudaMallocHost((void**)&pinnedC, bytes), "cudaMallocHost pinnedC");

        std::memcpy(pinnedA, a.data(), bytes);
        std::memcpy(pinnedB, b.data(), bytes);

        checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

        checkCuda(cudaMemcpyAsync(dA, pinnedA, bytes, cudaMemcpyHostToDevice, stream), "H2D dA");
        checkCuda(cudaMemcpyAsync(dB, pinnedB, bytes, cudaMemcpyHostToDevice, stream), "H2D dB");

        dim3 block(TILE, TILE);
        dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

        block_gemm_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, n);

        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaMemcpyAsync(pinnedC, dC, bytes, cudaMemcpyDeviceToHost, stream), "D2H dC");

        checkCuda(cudaStreamSynchronize(stream), "stream synchronize");

        std::vector<float> C(elems);
        std::memcpy(C.data(), pinnedC, bytes);

        checkCuda(cudaStreamDestroy(stream), "destroy stream");
        checkCuda(cudaFreeHost(pinnedA), "freeHost pinnedA");
        checkCuda(cudaFreeHost(pinnedB), "freeHost pinnedB");
        checkCuda(cudaFreeHost(pinnedC), "freeHost pinnedC");
        checkCuda(cudaFree(dA), "free dA");
        checkCuda(cudaFree(dB), "free dB");
        checkCuda(cudaFree(dC), "free dC");

        return C;
    } catch (...) {
        if (stream) cudaStreamDestroy(stream);
        if (pinnedA) cudaFreeHost(pinnedA);
        if (pinnedB) cudaFreeHost(pinnedB);
        if (pinnedC) cudaFreeHost(pinnedC);
        if (dA) cudaFree(dA);
        if (dB) cudaFree(dB);
        if (dC) cudaFree(dC);
        throw;
    }
}
