#include "naive_gemm_cuda.h"
#include <cuda.h>

__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float sum = 0.0f;
#pragma unroll 4
    for (int k = 0; k < n; ++k)
        sum += A[row * n + k] * B[k * n + col];
    C[row * n + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (a.empty() || b.empty()) return {};
    std::vector<float> c(n * n);

    float *A, *B, *C;
    cudaMallocManaged(&A, n * n * sizeof(float));
    cudaMallocManaged(&B, n * n * sizeof(float));
    cudaMallocManaged(&C, n * n * sizeof(float));

    memcpy(A, a.data(), n * n * sizeof(float));
    memcpy(B, b.data(), n * n * sizeof(float));

    dim3 block(32, 32);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    naive_gemm_kernel<<<grid, block>>>(A, B, C, n);
    cudaDeviceSynchronize();

    memcpy(c.data(), C, n * n * sizeof(float));

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return c;
}
