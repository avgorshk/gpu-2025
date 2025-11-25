#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <vector>

const int BLOCK_SIZE = 16;

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (a.size() != static_cast<size_t>(n * n) || b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Invalid matrix size");
    }

    std::vector<float> c(n * n, 0.0f);
    if (n == 0) return c;

    float* d_a, * d_b, * d_c;
    size_t size = n * n * sizeof(float);

    cudaError_t err;

    err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed for A");

    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error("cudaMalloc failed for B");
    }

    err = cudaMalloc(&d_c, size);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        throw std::runtime_error("cudaMalloc failed for C");
    }

    err = cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy failed for A");
    }

    err = cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy failed for B");
    }

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_multiply_kernel << <gridDim, blockDim >> > (d_a, d_b, d_c, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("Kernel launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("Device sync failed");
    }

    err = cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy failed for result");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}