#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

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
  cudaMalloc(&d_a, matrix_size);
  cudaMalloc(&d_b, matrix_size);
  cudaMalloc(&d_c, matrix_size);

  cudaMemcpy(d_a, a.data(), matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), matrix_size, cudaMemcpyHostToDevice);

  const float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, d_b, n, d_a, n,
              &beta, d_c, n);

  cudaMemcpy(c.data(), d_c, matrix_size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);

  return c;
}