#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstring> 
#include <iostream>

#define CUDA_CHECK(call)                                                        
    do {                                                                        
        cudaError_t _e = (call);                                                
        if (_e != cudaSuccess) {                                                
            std::string msg = "CUDA error: ";                                   
            msg += cudaGetErrorString(_e);                                      
            throw std::runtime_error(msg);                                      
        }                                                                       
    } while (0)

#ifndef TILE
#define TILE 16
#endif

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 4
#endif

template <typename idx_t>
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             idx_t N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    const unsigned tx = threadIdx.x; 
    const unsigned ty = threadIdx.y; 

    idx_t row_base = (idx_t)by * (idx_t)TILE + (idx_t)(ty);
    idx_t col = (idx_t)bx * (idx_t)TILE + (idx_t)tx;

    float acc[BLOCK_ROWS];
    #pragma unroll
    for (int r = 0; r < BLOCK_ROWS; ++r) acc[r] = 0.0f;

    for (idx_t m = 0; m < N; m += TILE) {
        #pragma unroll
        for (int r = 0; r < BLOCK_ROWS; ++r) {
            idx_t r_idx = row_base + (idx_t)(r * BLOCK_ROWS); 
            idx_t k_idx = m + (idx_t)tx;
            if (r_idx < N && k_idx < N) {
                // A[r_idx, k_idx]
                As[ty + r * BLOCK_ROWS][tx] = A[(size_t)r_idx * (size_t)N + (size_t)k_idx];
            } else {
                As[ty + r * BLOCK_ROWS][tx] = 0.0f;
            }
        }

        #pragma unroll
        for (int r = 0; r < BLOCK_ROWS; ++r) {
            idx_t kb = m + (idx_t)(ty + r * BLOCK_ROWS);
            if (kb < N && col < N) {
                // B[kb, col]
                Bs[ty + r * BLOCK_ROWS][tx] = B[(size_t)kb * (size_t)N + (size_t)col];
            } else {
                Bs[ty + r * BLOCK_ROWS][tx] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float bval = Bs[k][tx];
            #pragma unroll
            for (int r = 0; r < BLOCK_ROWS; ++r) {
                float aval = As[ty + r * BLOCK_ROWS][k];
                acc[r] += aval * bval;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < BLOCK_ROWS; ++r) {
        idx_t r_idx = row_base + (idx_t)(r * BLOCK_ROWS);
        if (r_idx < N && col < N) {
            C[(size_t)r_idx * (size_t)N + (size_t)col] = acc[r];
        }
    }
}

inline void launch_matmul(const float* d_A, const float* d_B, float* d_C, size_t N, cudaStream_t stream = 0) {
    dim3 block(TILE, BLOCK_ROWS);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    const size_t signed32_threshold = 46340; 
    if (N <= signed32_threshold) {
        matmul_tiled<uint32_t><<<grid, block, 0, stream>>>(d_A, d_B, d_C, static_cast<uint32_t>(N));
    } else {
        matmul_tiled<size_t><<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};
    size_t N = static_cast<size_t>(n);
    size_t elems = N * N;
    if (a.size() != elems || b.size() != elems) {
        throw std::invalid_argument("Matrix sizes must be n*n");
    }

    const size_t bytes = elems * sizeof(float);

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

        launch_matmul(d_A, d_B, d_C, N, stream);

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
