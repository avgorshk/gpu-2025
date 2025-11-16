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

#define CUDA_CHECK(call)                                                         
    do {                                                                         
        cudaError_t _e = (call);                                                 
        if (_e != cudaSuccess) {                                                 
            throw std::runtime_error(std::string("CUDA Error: ") +               
                                     cudaGetErrorString(_e));                    
        }                                                                        
    } while (0)

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

    float sum = 0.0f;

    const int numTiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        if (row < n && a_col < n)
            As[ty][tx] = A[row * n + a_col];
        else
            As[ty][tx] = 0.0f;

        if (b_row < n && col < n)
            Bs[ty][tx] = B[b_row * n + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};
    size_t N = static_cast<size_t>(n);
    size_t elems = N * N;
    if (a.size() != elems || b.size() != elems) {
        throw std::invalid_argument("Input matrices size mismatch (expected n*n).");
    }

    size_t bytes = elems * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    float *pinned_A = nullptr, *pinned_B = nullptr, *pinned_C = nullptr;
    cudaStream_t stream = nullptr;

    try {
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

        CUDA_CHECK(cudaMallocHost((void**)&pinned_A, bytes));
        CUDA_CHECK(cudaMallocHost((void**)&pinned_B, bytes));
        CUDA_CHECK(cudaMallocHost((void**)&pinned_C, bytes)); 

        std::memcpy(pinned_A, a.data(), bytes);
        std::memcpy(pinned_B, b.data(), bytes);

        CUDA_CHECK(cudaStreamCreate(&stream));

        CUDA_CHECK(cudaMemcpyAsync(d_A, pinned_A, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B, pinned_B, bytes, cudaMemcpyHostToDevice, stream));

        dim3 block(TILE, TILE);
        dim3 grid( (n + TILE - 1) / TILE, (n + TILE - 1) / TILE );

        block_gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, n);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(pinned_C, d_C, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> C(elems);
        std::memcpy(C.data(), pinned_C, bytes);

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFreeHost(pinned_A));
        CUDA_CHECK(cudaFreeHost(pinned_B));
        CUDA_CHECK(cudaFreeHost(pinned_C));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        return C;
    } catch (...) {
        if (stream) cudaStreamDestroy(stream);
        if (pinned_A) cudaFreeHost(pinned_A);
        if (pinned_B) cudaFreeHost(pinned_B);
        if (pinned_C) cudaFreeHost(pinned_C);
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        throw;
    }
}
