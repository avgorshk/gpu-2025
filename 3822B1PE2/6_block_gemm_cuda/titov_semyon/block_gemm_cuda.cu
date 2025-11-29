#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>

constexpr int BLOCK_SIZE = 16;

__global__ void block_gemm_kernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n) {

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockRow * BLOCK_SIZE + threadRow;
    int col = blockCol * BLOCK_SIZE + threadCol;

    float sum = 0.0f;
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int k = 0; k < numBlocks; k++) {
        int aRow = blockRow * BLOCK_SIZE + threadRow;
        int aCol = k * BLOCK_SIZE + threadCol;
        if (aRow < n && aCol < n) {
            As[threadRow][threadCol] = A[aRow * n + aCol];
        }
        else {
            As[threadRow][threadCol] = 0.0f;
        }
        int bRow = k * BLOCK_SIZE + threadRow;
        int bCol = blockCol * BLOCK_SIZE + threadCol;
        if (bRow < n && bCol < n) {
            Bs[threadRow][threadCol] = B[bRow * n + bCol];
        }
        else {
            Bs[threadRow][threadCol] = 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += As[threadRow][i] * Bs[i][threadCol];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

__global__ void block_gemm_optimized_kernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n) {

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    constexpr int TILE_SIZE = 2;
    float sum[TILE_SIZE][TILE_SIZE] = { 0.0f };

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int k = 0; k < numBlocks; k++) {
        for (int ti = 0; ti < TILE_SIZE; ti++) {
            int aRow = blockRow * BLOCK_SIZE + threadRow * TILE_SIZE + ti;
            int aCol = k * BLOCK_SIZE + threadCol;
            if (aRow < n && aCol < n) {
                As[threadRow * TILE_SIZE + ti][threadCol] = A[aRow * n + aCol];
            }
            else {
                As[threadRow * TILE_SIZE + ti][threadCol] = 0.0f;
            }
        }

        for (int tj = 0; tj < TILE_SIZE; tj++) {
            int bRow = k * BLOCK_SIZE + threadRow;
            int bCol = blockCol * BLOCK_SIZE + threadCol * TILE_SIZE + tj;
            if (bRow < n && bCol < n) {
                Bs[threadRow][threadCol * TILE_SIZE + tj] = B[bRow * n + bCol];
            }
            else {
                Bs[threadRow][threadCol * TILE_SIZE + tj] = 0.0f;
            }
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int ti = 0; ti < TILE_SIZE; ti++) {
                for (int tj = 0; tj < TILE_SIZE; tj++) {
                    sum[ti][tj] += As[threadRow * TILE_SIZE + ti][i] *
                        Bs[i][threadCol * TILE_SIZE + tj];
                }
            }
        }

        __syncthreads();
    }

    for (int ti = 0; ti < TILE_SIZE; ti++) {
        for (int tj = 0; tj < TILE_SIZE; tj++) {
            int row = blockRow * BLOCK_SIZE + threadRow * TILE_SIZE + ti;
            int col = blockCol * BLOCK_SIZE + threadCol * TILE_SIZE + tj;
            if (row < n && col < n) {
                C[row * n + col] = sum[ti][tj];
            }
        }
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Matrix sizes do not match n*n");
    }

    size_t bytes = n * n * sizeof(float);
    std::vector<float> c(n * n);

    if (n == 0) return c;

    float* d_a = nullptr, * d_b = nullptr, * d_c = nullptr;

    checkCudaError(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    checkCudaError(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "cudaStreamCreate");

    checkCudaError(cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream), "Copy a to device");
    checkCudaError(cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream), "Copy b to device");

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    block_gemm_kernel << <gridDim, blockDim, 0, stream >> > (d_a, d_b, d_c, n);

    checkCudaError(cudaGetLastError(), "Kernel launch");

    checkCudaError(cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream), "Copy c from device");

    checkCudaError(cudaStreamSynchronize(stream), "Stream synchronization");

    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}