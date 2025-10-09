#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void gemm_block_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float shared_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float sum = 0.0f;
    int block_count = n / BLOCK_SIZE;

    for (int bk = 0; bk < block_count; ++bk) {

        int a_row = by * BLOCK_SIZE + ty;
        int a_col = bk * BLOCK_SIZE + tx;
        if (a_row < n && a_col < n) {
            shared_a[ty][tx] = a[a_row * n + a_col];
        } else {
            shared_a[ty][tx] = 0.0f;
        }

        int b_row = bk * BLOCK_SIZE + ty;
        int b_col = bx * BLOCK_SIZE + tx;
        if (b_row < n && b_col < n) {
            shared_b[ty][tx] = b[b_row * n + b_col];
        } else {
            shared_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += shared_a[ty][k] * shared_b[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
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
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    checkCudaError(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    checkCudaError(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    checkCudaError(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "Copy a to device");
    checkCudaError(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "Copy b to device");

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_block_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    std::vector<float> c(n * n);
    checkCudaError(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "Copy c from device");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}