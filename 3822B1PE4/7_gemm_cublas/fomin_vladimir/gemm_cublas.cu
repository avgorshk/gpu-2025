#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b, int n)
{
  if (a.size() != static_cast<size_t>(n * n) ||
      b.size() != static_cast<size_t>(n * n))
  {
    throw std::invalid_argument("Invalid matrix dimensions");
  }

  std::vector<float> c(n * n);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  cublasHandle_t handle;
  cublasCreate(&handle);

  size_t matrix_size = n * n * sizeof(float);

  cudaError_t cudaStatus;
  cudaStatus = cudaMalloc(&d_a, matrix_size);
  if (cudaStatus != cudaSuccess)
  {
    throw std::runtime_error("Failed to allocate device memory for d_a");
  }

  cudaStatus = cudaMalloc(&d_b, matrix_size);
  if (cudaStatus != cudaSuccess)
  {
    cudaFree(d_a);
    throw std::runtime_error("Failed to allocate device memory for d_b");
  }

  cudaStatus = cudaMalloc(&d_c, matrix_size);
  if (cudaStatus != cudaSuccess)
  {
    cudaFree(d_a);
    cudaFree(d_b);
    throw std::runtime_error("Failed to allocate device memory for d_c");
  }

  cudaStatus = cudaMemcpy(d_a, a.data(), matrix_size, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
  {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    throw std::runtime_error("Failed to copy matrix a to device");
  }

  cudaStatus = cudaMemcpy(d_b, b.data(), matrix_size, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
  {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    throw std::runtime_error("Failed to copy matrix b to device");
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasStatus_t cublasStatus = cublasSgemm(handle,
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            n, n, n,
                                            &alpha,
                                            d_b,
                                            n,
                                            d_a,
                                            n,
                                            &beta,
                                            d_c,
                                            n);

  if (cublasStatus != CUBLAS_STATUS_SUCCESS)
  {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    throw std::runtime_error("CUBLAS gemm operation failed");
  }

  cudaStatus = cudaMemcpy(c.data(), d_c, matrix_size, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
  {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    throw std::runtime_error("Failed to copy result to host");
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);

  return c;
}