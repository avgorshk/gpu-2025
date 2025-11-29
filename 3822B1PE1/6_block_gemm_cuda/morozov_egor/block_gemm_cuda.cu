#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

constexpr int BLOCK_SIZE = 16;
constexpr int TILE_SIZE = BLOCK_SIZE;
using uint = unsigned;

__global__ void blockMatMulKernel(const float *A, const float *B, float *C, int n) {
    uint i0 = blockIdx.y * TILE_SIZE + threadIdx.y;
    uint j0 = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    __shared__ float A_[TILE_SIZE][TILE_SIZE];
    __shared__ float B_[TILE_SIZE][TILE_SIZE];

    for (int i = 0; i < (n + TILE_SIZE - 1) / TILE_SIZE; ++i) {
        uint ai = i * TILE_SIZE + threadIdx.x;
        uint bj = i * TILE_SIZE + threadIdx.y;
        if (i0 < n && ai < n) {
            A_[threadIdx.y][threadIdx.x] = A[i0 * n + ai];
        } else {
            A_[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (bj < n && j0 < n) {
            B_[threadIdx.y][threadIdx.x] = B[bj * n + j0];
        } else {
            B_[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_[threadIdx.y][k] * B_[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (i0 < n && j0 < n) {
        C[i0 * n + j0] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n) {
    size_t size = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE,
                 (n + TILE_SIZE - 1) / TILE_SIZE);

    blockMatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    std::vector<float> result(n * n);
    CUDA_CHECK(cudaMemcpy(result.data(), d_C, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}