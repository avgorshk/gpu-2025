#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "block_gemm_cuda.h"

#define TILE 16

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int n) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;

  int numTiles = (n + TILE - 1) / TILE;

  for (int t = 0; t < numTiles; ++t) {
    int a_col = t * TILE + threadIdx.x;
    int b_row = t * TILE + threadIdx.y;

    if (row < n && a_col < n) {
      As[threadIdx.y][threadIdx.x] = A[row * n + a_col];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (b_row < n && col < n) {
      Bs[threadIdx.y][threadIdx.x] = B[b_row * n + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  const int nn = n * n;
  std::vector<float> c(nn, 0.0f);

  if (n <= 0 || (int)a.size() != nn || (int)b.size() != nn) {
    return c;
  }

  static float* d_A = nullptr;
  static float* d_B = nullptr;
  static float* d_C = nullptr;
  static int capacity_n = 0;

  if (n > capacity_n) {
    if (d_A) {
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);
    }

    size_t bytes_alloc = static_cast<size_t>(n) * n * sizeof(float);
    if (cudaMalloc(&d_A, bytes_alloc) != cudaSuccess ||
        cudaMalloc(&d_B, bytes_alloc) != cudaSuccess ||
        cudaMalloc(&d_C, bytes_alloc) != cudaSuccess) {
      d_A = d_B = d_C = nullptr;
      capacity_n = 0;
      throw std::runtime_error("cudaMalloc failed in BlockGemmCUDA");
    }
    capacity_n = n;
  }

  size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

  cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

  dim3 block(TILE, TILE);
  dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

  block_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("Kernel launch failed in BlockGemmCUDA");
  }

  cudaDeviceSynchronize();

  cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

  return c;
}
