#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void naive_gemm_kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0f;
        int k;
        for (k = 0; k + 4 <= n; k += 4) {
            value += A[row * n + k] * B[k * n + col];
            value += A[row * n + k + 1] * B[(k + 1) * n + col];
            value += A[row * n + k + 2] * B[(k + 2) * n + col];
            value += A[row * n + k + 3] * B[(k + 3) * n + col];
        }
        for (; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = value;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (a.empty() || b.empty() || n <= 0) return {};

    size_t bytes = n * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_A, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    naive_gemm_kernel << <grid, block, 0, stream >> > (d_A, d_B, d_C, n);

    cudaMemcpyAsync(c.data(), d_C, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}