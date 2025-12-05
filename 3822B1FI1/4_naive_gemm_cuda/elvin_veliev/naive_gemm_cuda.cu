#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

__global__ void kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0f;
        for (int k = 0; k < n; ++k) {
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

    kernel <<<grid, block, 0, stream>>> (d_A, d_B, d_C, n);

    cudaMemcpyAsync(c.data(), d_C, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}