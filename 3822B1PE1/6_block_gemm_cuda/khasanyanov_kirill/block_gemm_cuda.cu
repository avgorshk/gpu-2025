#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void blockGemmKernel(const float *a, const float *b, float *c,
                                int n) {
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  float sum = 0.0f;

  int globalRow = blockRow * BLOCK_SIZE + row;
  int globalCol = blockCol * BLOCK_SIZE + col;

  int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int m = 0; m < numBlocks; ++m) {
    int aRow = globalRow;
    int aCol = m * BLOCK_SIZE + col;

    if (aRow < n && aCol < n) {
      As[row][col] = a[aRow * n + aCol];
    } else {
      As[row][col] = 0.0f;
    }

    int bRow = m * BLOCK_SIZE + row;
    int bCol = globalCol;

    if (bRow < n && bCol < n) {
      Bs[row][col] = b[bRow * n + bCol];
    } else {
      Bs[row][col] = 0.0f;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      sum += As[row][k] * Bs[k][col];
    }
    __syncthreads();
  }

  if (globalRow < n && globalCol < n) {
    c[globalRow * n + globalCol] = sum;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b, int n) {
  if (n == 0) {
    return std::vector<float>();
  }

  std::vector<float> c(n * n);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;

  size_t bytes = n * n * sizeof(float);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  blockGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return c;
}
