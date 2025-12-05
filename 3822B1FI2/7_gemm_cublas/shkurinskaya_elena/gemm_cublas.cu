#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
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

  static cublasHandle_t handle = nullptr;
  static cudaStream_t stream = nullptr;

  if (!handle) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasCreate failed");
    }
  }
  if (!stream) {
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      throw std::runtime_error("cudaStreamCreate failed");
    }

    cublasSetStream(handle, stream);
  }

  if (n > capacity_n) {
    if (d_A) {
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);
    }

    size_t bytes_alloc = (size_t)n * n * sizeof(float);
    if (cudaMalloc(&d_A, bytes_alloc) != cudaSuccess ||
        cudaMalloc(&d_B, bytes_alloc) != cudaSuccess ||
        cudaMalloc(&d_C, bytes_alloc) != cudaSuccess) {
      d_A = d_B = d_C = nullptr;
      capacity_n = 0;
      throw std::runtime_error("cudaMalloc failed in GemmCUBLAS");
    }
    capacity_n = n;
  }

  size_t bytes = (size_t)n * n * sizeof(float);

  cudaMemcpyAsync(d_A, a.data(), bytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_B, b.data(), bytes, cudaMemcpyHostToDevice, stream);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  int m = n;
  int k = n;
  int n_cols = n;

  cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n_cols,
                                    k, &alpha, d_B, m, d_A, k, &beta, d_C, m);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublasSgemm failed");
  }

  cudaMemcpyAsync(c.data(), d_C, bytes, cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  return c;
}
