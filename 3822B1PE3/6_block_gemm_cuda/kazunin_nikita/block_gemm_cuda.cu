#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iostream>

#define CHECK_CUDA(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        throw std::runtime_error(cudaGetErrorString(err)); \
    }                                                      \
} while(0)

constexpr int TILE = 16;

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = blockRow * TILE + threadIdx.y;
    int col = blockCol * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int m = 0; m < n; m += TILE) {
        int a_col = m + threadIdx.x;
        if (row < n && a_col < n) {
            sA[threadIdx.y][threadIdx.x] = A[row * n + a_col];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = m + threadIdx.y;
        if (b_row < n && col < n) {
            sB[threadIdx.y][threadIdx.x] = B[b_row * n + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
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
    if (a.size() != static_cast<size_t>(n) * n || b.size() != static_cast<size_t>(n) * n) {
        throw std::invalid_argument("Input sizes must be n*n");
    }

    size_t elems = static_cast<size_t>(n) * n;
    size_t bytes = elems * sizeof(float);
    std::vector<float> out(elems);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));

    float *hA = nullptr, *hB = nullptr, *hC = nullptr;
    CHECK_CUDA(cudaMallocHost(&hA, bytes));
    CHECK_CUDA(cudaMallocHost(&hB, bytes));
    CHECK_CUDA(cudaMallocHost(&hC, bytes));

    std::memcpy(hA, a.data(), bytes);
    std::memcpy(hB, b.data(), bytes);

    CHECK_CUDA(cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(dB, hB, bytes, cudaMemcpyHostToDevice, stream));

    dim3 blockDim(TILE, TILE);
    dim3 gridDim((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    block_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(dA, dB, dC, n);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(hC, dC, bytes, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::memcpy(out.data(), hC, bytes);

    CHECK_CUDA(cudaFreeHost(hA));
    CHECK_CUDA(cudaFreeHost(hB));
    CHECK_CUDA(cudaFreeHost(hC));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return out;
}
