#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <cassert>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define B_PAD (BLOCK_SIZE + 1)

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n,
                                  int numTiles) {
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int tileRow = blockIdx.y;
    const int tileCol = blockIdx.x;

    const int row = tileRow * BLOCK_SIZE + ty;
    const int col = tileCol * BLOCK_SIZE + tx;

    __shared__ float a_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_tile[BLOCK_SIZE][B_PAD];

    float sum = 0.0f;

    for (int t = 0; t < numTiles; ++t) {
        const int a_col_base = t * BLOCK_SIZE;
        const int b_row_base = t * BLOCK_SIZE;

        int a_row_idx = row;
        if (a_row_idx < n) {
            int col4 = a_col_base + (tx & ~3);
            if (((tx & 3) == 0) && (col4 + 3) < n) {
                const float4* src = reinterpret_cast<const float4*>(&A[a_row_idx * n + col4]);
                float4 v = *src;
                int base_local = tx & ~3;
                a_tile[ty][base_local + 0] = v.x;
                a_tile[ty][base_local + 1] = v.y;
                a_tile[ty][base_local + 2] = v.z;
                a_tile[ty][base_local + 3] = v.w;
            } else {
                int a_col = a_col_base + tx;
                a_tile[ty][tx] = (a_col < n) ? A[a_row_idx * n + a_col] : 0.0f;
            }
        } else {
            a_tile[ty][tx] = 0.0f;
        }

        int b_row = b_row_base + ty;
        if (b_row < n) {
            int b_col_idx = tileCol * BLOCK_SIZE + tx;
            int b_col4 = (b_col_idx & ~3);
            if (((tx & 3) == 0) && (b_col4 + 3) < n) {
                const float4* src = reinterpret_cast<const float4*>(&B[b_row * n + b_col4]);
                float4 v = *src;
                int base_local = tx & ~3;
                b_tile[ty][base_local + 0] = v.x;
                b_tile[ty][base_local + 1] = v.y;
                b_tile[ty][base_local + 2] = v.z;
                b_tile[ty][base_local + 3] = v.w;
            } else {
                int b_col = tileCol * BLOCK_SIZE + tx;
                b_tile[ty][tx] = (b_col < n) ? B[b_row * n + b_col] : 0.0f;
            }
        } else {
            b_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += a_tile[ty][k] * b_tile[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if ((int)a.size() != n * n || (int)b.size() != n * n)
        throw std::invalid_argument("Input matrices must be size n*n");

    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);

    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaHostRegister(const_cast<float*>(a.data()), bytes, cudaHostRegisterDefault);
    cudaHostRegister(const_cast<float*>(b.data()), bytes, cudaHostRegisterDefault);
    cudaHostRegister(c.data(),                     bytes, cudaHostRegisterDefault);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpyAsync(d_A, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    const int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    block_gemm_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, n, numTiles);

    cudaMemcpyAsync(c.data(), d_C, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaHostUnregister(const_cast<float*>(a.data()));
    cudaHostUnregister(const_cast<float*>(b.data()));
    cudaHostUnregister(c.data());

    cudaStreamDestroy(stream);

    return c;
}
