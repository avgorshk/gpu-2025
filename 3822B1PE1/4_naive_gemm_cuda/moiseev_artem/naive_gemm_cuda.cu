#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

__global__ void gemm_naive_kernel(const float* a, const float* b, float* c, int n) {
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

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
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

    checkCudaError(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    checkCudaError(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    checkCudaError(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "Copy a to device");
    checkCudaError(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "Copy b to device");

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);

    gemm_naive_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    std::vector<float> c(n * n);
    checkCudaError(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "Copy c from device");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}