#include <cuda_runtime.h>

#include <vector>

#include "naive_gemm_cuda.h"

constexpr int TILE_SIZE = 64;
constexpr int SUB_TILE_SIZE = 4;

__global__ void gemmKernel(const float* a, const float* b, float* o, int n) {
  __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
  __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  float sums[SUB_TILE_SIZE][SUB_TILE_SIZE] = {0};

  int row = by * TILE_SIZE + ty * SUB_TILE_SIZE;
  int col = bx * TILE_SIZE + tx * SUB_TILE_SIZE;

  int tiles_count = (n + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < tiles_count; t++) {
    for (int i = 0; i < SUB_TILE_SIZE; i++) {
      for (int j = 0; j < SUB_TILE_SIZE; j++) {
        int l_row = ty * SUB_TILE_SIZE + i;
        int l_col = tx * SUB_TILE_SIZE + j;

        int a_row = row + i;
        int a_col = t * TILE_SIZE + l_col;
        int b_row = t * TILE_SIZE + l_row;
        int b_col = col + j;

        if (a_row < n && a_col < n)
          sharedA[l_row][l_col] = a[a_row * n + a_col];
        else
          sharedA[l_row][l_col] = 0.0f;

        if (b_row < n && b_col < n)
          sharedB[l_row][l_col] = b[b_row * n + b_col];
        else
          sharedB[l_row][l_col] = 0.0f;
      }
    }

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; k++) {
      for (int i = 0; i < SUB_TILE_SIZE; i++) {
        for (int j = 0; j < SUB_TILE_SIZE; j++) {
          sums[i][j] += sharedA[ty * SUB_TILE_SIZE + i][k] *
                        sharedB[k][tx * SUB_TILE_SIZE + j];
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < SUB_TILE_SIZE; i++) {
    for (int j = 0; j < SUB_TILE_SIZE; j++) {
      if (row + i < n && col + j < n) {
        o[(row + i) * n + (col + j)] = sums[i][j];
      }
    }
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  std::vector<float> c(n * n);

  float *inputA, *inputB, *output;
  size_t size = n * n * sizeof(float);

  cudaMalloc(&inputA, size);
  cudaMalloc(&inputB, size);
  cudaMalloc(&output, size);

  cudaMemcpy(inputA, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(inputB, b.data(), size, cudaMemcpyHostToDevice);

  dim3 blockSize(TILE_SIZE / SUB_TILE_SIZE, TILE_SIZE / SUB_TILE_SIZE);
  dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE,
                (n + TILE_SIZE - 1) / TILE_SIZE);

  gemmKernel<<<gridSize, blockSize>>>(inputA, inputB, output, n);

  cudaDeviceSynchronize();

  cudaMemcpy(c.data(), output, size, cudaMemcpyDeviceToHost);

  cudaFree(inputA);
  cudaFree(inputB);
  cudaFree(output);

  return c;
}
