#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "block_gemm_cuda.h"

#ifndef TILE
#define TILE 16
#endif

__global__ void gemm_block_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int n) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  __shared__ float Cs[TILE][TILE];

  const int ty = threadIdx.y;
  const int tx = threadIdx.x;

  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.0f;
  Cs[ty][tx] = 0.0f;

  for (int t = 0; t < n; t += TILE) {
    if (row < n && (t + tx) < n)
      As[ty][tx] = A[row * n + (t + tx)];
    else
      As[ty][tx] = 0.0f;

    if ((t + ty) < n && col < n)
      Bs[ty][tx] = B[(t + ty) * n + col];
    else
      Bs[ty][tx] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }

    Cs[ty][tx] += acc;
    acc = 0.0f;

    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = Cs[ty][tx];
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  size_t bytes = n * n * sizeof(float);
  float *dA, *dB, *dC;

  cudaMalloc(&dA, bytes);
  cudaMalloc(&dB, bytes);
  cudaMalloc(&dC, bytes);

  cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

  gemm_block_kernel<<<grid, block>>>(dA, dB, dC, n);
  cudaDeviceSynchronize();

  std::vector<float> c(n * n);
  cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return c;
}
