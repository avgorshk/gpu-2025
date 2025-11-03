#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

__global__ void naive_gemm_kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0f;
        int k = 0;
        // Loop unrolling
        for (; k + 3 < n; k += 4) {
            value += A[row * n + k] * B[k * n + col];
            value += A[row * n + k + 1] * B[(k + 1) * n + col];
            value += A[row * n + k + 2] * B[(k + 2) * n + col];
            value += A[row * n + k + 3] * B[(k + 3) * n + col];
        }
        // Handle remaining elements
        for (; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (a.empty() || b.empty() || n <= 0) {
        return {};
    }

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> c(static_cast<size_t>(n) * n, 0.0f);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpyAsync(d_A, a.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, b.data(), bytes, cudaMemcpyHostToDevice, stream));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    naive_gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(c.data(), d_C, bytes, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return c;
}
