#include "block_gemm_cuda.h"

template <int BLOCK_SIZE>
__global__ void BlockGemmKernel(const float* a, const float* b, float* o,
                                int n) {
  const int BUFFERS = 2;
  __shared__ float abl[BLOCK_SIZE][BLOCK_SIZE][BUFFERS];
  __shared__ float bbl[BLOCK_SIZE][BLOCK_SIZE][BUFFERS];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;

  const int row = blockRow * BLOCK_SIZE + ty;
  const int col = blockCol * BLOCK_SIZE + tx;

  float sum = 0.0f;

  const int blocks = n / BLOCK_SIZE;

  const int a_row = blockRow * BLOCK_SIZE + ty;
  const int b_col = blockCol * BLOCK_SIZE + tx;

  float locBuf[BLOCK_SIZE][BUFFERS];

  for (int m = 0; m < blocks; m += BUFFERS) {
    for (int i = 0; i < BUFFERS; i++) {
      const int a_col = (m + i) * BLOCK_SIZE + tx;
      abl[ty][tx][i] = a[a_row * n + a_col];
      // tried insert memcpy here but got more than 64 registers
      // it was 2 loops for abl and bbl, sync&memcpy in between
      const int b_row = (m + i) * BLOCK_SIZE + ty;
      bbl[ty][tx][i] = b[b_row * n + b_col];
    }

    __syncthreads();

    memcpy(locBuf, abl[ty], sizeof(locBuf));
    for (int k = 0; k < BLOCK_SIZE; k++) {
      // spends too much time due to MIO stall
      sum += locBuf[k][1] * bbl[k][tx][1];
      sum += locBuf[k][0] * bbl[k][tx][0];  
    }

    __syncthreads();
  }

  o[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  std::vector<float> c(n * n);

  float *inputA, *inputB, *output;
  size_t size = n * n * sizeof(float);

  cudaMalloc(&inputA, size);
  cudaMalloc(&inputB, size);
  cudaMalloc(&output, size);

  cudaMemcpy(inputA, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(inputB, b.data(), size, cudaMemcpyHostToDevice);

  const int BLOCK_SIZE = 32;

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  BlockGemmKernel<BLOCK_SIZE><<<gridDim, blockDim>>>(inputA, inputB, output, n);

  cudaMemcpy(c.data(), output, size, cudaMemcpyDeviceToHost);

  cudaFree(inputA);
  cudaFree(inputB);
  cudaFree(output);

  return c;
}