#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    float sum = 0.0f;

#pragma unroll 4
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }

    C[row * n + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (a.empty() || b.empty()) return {};

    std::vector<float> c(n * n, 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void**)&d_A, n * n * sizeof(float));
    cudaMalloc((void**)&d_B, n * n * sizeof(float));
    cudaMalloc((void**)&d_C, n * n * sizeof(float));

    float *hA_pinned = nullptr, *hB_pinned = nullptr, *hC_pinned = nullptr;
    cudaMallocHost((void**)&hA_pinned, n * n * sizeof(float));
    cudaMallocHost((void**)&hB_pinned, n * n * sizeof(float));
    cudaMallocHost((void**)&hC_pinned, n * n * sizeof(float));

    memcpy(hA_pinned, a.data(), n * n * sizeof(float));
    memcpy(hB_pinned, b.data(), n * n * sizeof(float));

    cudaMemcpyAsync(d_A, hA_pinned, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, hB_pinned, n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, n);

    cudaMemcpyAsync(hC_pinned, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    memcpy(c.data(), hC_pinned, n * n * sizeof(float));

    cudaFreeHost(hA_pinned);
    cudaFreeHost(hB_pinned);
    cudaFreeHost(hC_pinned);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}
