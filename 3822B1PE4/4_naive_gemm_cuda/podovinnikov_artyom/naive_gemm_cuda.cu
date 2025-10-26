#include <cuda_runtime.h>

#include <iostream>

#include "naive_gemm_cuda.h"

__global__ void gemm_kernel(const float* A, const float* B, float* C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0.0f;

    for (int k = 0; k < n; ++k) {
      sum += A[row * n + k] * B[k * n + col];
    }

    C[row * n + col] = sum;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  size_t bytes = n * n * sizeof(float);

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;


  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);


  cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);


  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);


  gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, n);


  cudaDeviceSynchronize();

 
  std::vector<float> c(n * n);
  cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);


  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return c;
} //
