#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <iostream>

__global__ void gemm_naive_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Matrix sizes do not match n*n");
    }

    size_t bytes = n * n * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    if (cudaMalloc(&d_a, bytes) != cudaSuccess ||
        cudaMalloc(&d_b, bytes) != cudaSuccess ||
        cudaMalloc(&d_c, bytes) != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed");
    }

    if (cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("CUDA memcpy (H2D) failed");
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);

    gemm_naive_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("CUDA kernel execution failed");
    }

    std::vector<float> c(n * n);
    if (cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("CUDA memcpy (D2H) failed");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}

