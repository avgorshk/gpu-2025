#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "naive_gemm_cuda.h"

__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= n || col >= n) return;

  float sum = 0.0f;

  int k = 0;
  int limit = n & ~3;

  for (; k < limit; k += 4) {
    float a0 = A[row * n + (k)];
    float a1 = A[row * n + (k + 1)];
    float a2 = A[row * n + (k + 2)];
    float a3 = A[row * n + (k + 3)];

    float b0 = B[(k)*n + col];
    float b1 = B[(k + 1) * n + col];
    float b2 = B[(k + 2) * n + col];
    float b3 = B[(k + 3) * n + col];

    sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
  }

  for (; k < n; ++k) {
    sum += A[row * n + k] * B[k * n + col];
  }

  C[row * n + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  const int nn = n * n;
  std::vector<float> c(nn);

  if (n <= 0 || a.size() != (size_t)nn || b.size() != (size_t)nn) {
    return c;
  }

  static float* d_a = nullptr;
  static float* d_b = nullptr;
  static float* d_c = nullptr;
  static int capacity_n = 0;

  if (n > capacity_n) {
    if (d_a) {
      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
    }

    size_t bytes = (size_t)n * (size_t)n * sizeof(float);

    if (cudaMalloc(&d_a, bytes) != cudaSuccess ||
        cudaMalloc(&d_b, bytes) != cudaSuccess ||
        cudaMalloc(&d_c, bytes) != cudaSuccess) {
      d_a = d_b = d_c = nullptr;
      capacity_n = 0;
      throw std::runtime_error("cudaMalloc failed in NaiveGemmCUDA");
    }
    capacity_n = n;
  }

  size_t bytes = (size_t)n * (size_t)n * sizeof(float);

  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

  naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("Kernel launch failed in NaiveGemmCUDA");
  }

  cudaDeviceSynchronize();

  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  return c;
}
