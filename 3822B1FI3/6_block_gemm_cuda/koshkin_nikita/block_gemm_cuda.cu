#include "block_gemm_cuda.h"

constexpr int BLOCK_SIZE = 16;

__global__ void BlockGemmKernel(float* _A, float* _B, float* _C, size_t _n) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float Ashr[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bshr[BLOCK_SIZE][BLOCK_SIZE];

  float res = 0.f;

  for (int subblock = 0; subblock < _n; subblock += BLOCK_SIZE) {
    if (y < _n && subblock + threadIdx.x < _n)
      Ashr[threadIdx.y][threadIdx.x] = _A[y * _n + (subblock + threadIdx.x)];
    else
      Ashr[threadIdx.y][threadIdx.x] = 0.f;

    if (x < _n && subblock + threadIdx.y < _n)
      Bshr[threadIdx.y][threadIdx.x] = _B[(subblock + threadIdx.y) * _n + x];
    else
      Bshr[threadIdx.y][threadIdx.x] = 0.f;

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      res += Ashr[threadIdx.y][k] * Bshr[k][threadIdx.x];
    }

    __syncthreads();
  }
 
  if (y < _n && x < _n) {
    _C[y * _n + x] = res;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {
  size_t size = n * n * sizeof(float);
  float* A, * B, * C;
  std::vector<float> result(n * n, 0.f);

  cudaMalloc(&A, size);
  cudaMalloc(&B, size);
  cudaMalloc(&C, size);

  cudaMemcpy(A, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(B, b.data(), size, cudaMemcpyHostToDevice);

  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 num_blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  BlockGemmKernel << <num_blocks, block_dim >> > (A, B, C, n);

  cudaMemcpy(result.data(), C, size, cudaMemcpyDeviceToHost);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return result;
}