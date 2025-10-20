#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= n || y >= n) {
    return;
  }

  b = &b[x];
  const float4* a4 = reinterpret_cast<const float4*>(&a[y*n]);

  for (size_t k = 0; k < (n >> 2); ++k) {
    float4 a = a4[k];

    c[y * n + x] += a.x * b[k * 4 * n] 
                  + a.y * b[(k * 4 + 1) * n] 
                  + a.z * b[(k * 4 + 2) * n] 
                  + a.w * b[(k * 4 + 3) * n];
  }
}

std::vector<float> NaiveGemmCUDA(
  const std::vector<float>& a,
  const std::vector<float>& b,
  int n
) {
  size_t size = n * n * sizeof(float);
  float *device_a, *device_b, *device_c;
  std::vector<float> c(n * n, 0);

  cudaMalloc(&device_a, size);
  cudaMalloc(&device_b, size);
  cudaMalloc(&device_c, size);

  cudaMemcpy(device_a, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b.data(), size, cudaMemcpyHostToDevice);

  constexpr int block_size = 32;
  dim3 block_dim(block_size, block_size);
  dim3 num_blocks((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

  kernel<<<num_blocks, block_dim>>>(device_a, device_b, device_c, n);

  cudaMemcpy(c.data(), device_c, size, cudaMemcpyDeviceToHost);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return c;
}
