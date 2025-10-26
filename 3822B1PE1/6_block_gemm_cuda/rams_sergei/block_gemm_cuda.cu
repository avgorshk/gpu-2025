#include "block_gemm_cuda.h"
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void kernel(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, size_t n) {
  size_t x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  size_t y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  if (x >= n || y >= n) {
    return;
  }
  
  __shared__ float ashared[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float bshared[BLOCK_SIZE][BLOCK_SIZE];

  for (size_t block = 0; block < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; block++) {
    ashared[threadIdx.y][threadIdx.x] = a[y * n + (block * BLOCK_SIZE + threadIdx.x)];
    bshared[threadIdx.y][threadIdx.x] = b[x + n * (block * BLOCK_SIZE + threadIdx.y)];

    __syncthreads();
    for (size_t k = 0; k < BLOCK_SIZE; k++) {
      c[y * n + x] += ashared[threadIdx.y][k] * bshared[k][threadIdx.x];
    }
    __syncthreads();
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
  size_t size = n * n * sizeof(float);
  float *device_a, *device_b, *device_c;
  std::vector<float> c(n * n, 0);

  cudaMalloc(&device_a, size);
  cudaMalloc(&device_b, size);
  cudaMalloc(&device_c, size);

  cudaMemcpy(device_a, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b.data(), size, cudaMemcpyHostToDevice);

  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 num_blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  kernel<<<num_blocks, block_dim>>>(device_a, device_b, device_c, n);

  cudaMemcpy(c.data(), device_c, size, cudaMemcpyDeviceToHost);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return c;
}
