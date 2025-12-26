#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void BlockGemmKernel(const float* A,
  const float* B,
  float* C,
  int n) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  float sum = 0.0f;

  for (int bk = 0; bk < n; bk += BLOCK_SIZE) {
    int a_col = bk + tx;
    int b_row = bk + ty;

    if (row < n && a_col < n) As[ty][tx] = A[row * n + a_col];
    else As[ty][tx] = 0.0f;

    if (b_row < n && col < n) Bs[ty][tx] = B[b_row * n + col];
    else Bs[ty][tx] = 0.0f;

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
      sum += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {
  std::vector<float> c(n * n);
  size_t size = n * n * sizeof(float);

  float* d_A, * d_B, * d_C;

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
    (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  BlockGemmKernel << <grid, block >> > (d_A, d_B, d_C, n);

  cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return c;
}