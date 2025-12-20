#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

constexpr int TILE = 16;

__global__ void gemm_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < n; t += TILE) {
        if (row < n && t + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && t + threadIdx.y < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> c(n * n);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, n);

    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}
