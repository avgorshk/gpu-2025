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

  // Копируем матрицы на устройство (как они есть, в row-major)
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

  // ВАЖНО: cuBLAS ожидает column-major матрицы
  // Мы хотим вычислить C = A * B (row-major)
  // В column-major это эквивалентно C^T = B^T * A^T
  
  // Используем транспонирование для обеих матриц
  // C_col_major = alpha * A_col_major^T * B_col_major^T + beta * C_col_major
  // Но поскольку мы передаем матрицы в row-major, они интерпретируются как транспонированные
  
  // Правильный вызов для вычисления C_row_major = A_row_major * B_row_major через cuBLAS:
  // C_col_major = (A_row_major * B_row_major)^T = B_row_major^T * A_row_major^T
  // = B_col_major * A_col_major (поскольку B_row_major^T = B_col_major)
  
  cublasStatus_t cublasStatus = cublasSgemm(handle,
                                            CUBLAS_OP_N, // B не транспонировать (уже в нужном формате)
                                            CUBLAS_OP_N, // A не транспонировать
                                            n, n, n,
                                            &alpha,
                                            d_b, // B в памяти (row-major, но для cuBLAS это column-major B^T)
                                            n,
                                            d_a, // A в памяти (row-major, но для cuBLAS это column-major A^T)
                                            n,
                                            &beta,
                                            d_c, // C в памяти (будет column-major C^T)
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

  // Полученный результат в памяти - это C^T в column-major,
  // что эквивалентно C в row-major (транспонирование матрицы, представленной в другом порядке)
  // Так что дополнительное транспонирование не нужно

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);

  return c;
}