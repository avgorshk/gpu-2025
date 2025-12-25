#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// Implementation of GEMM using cuBLAS
// Performs C = A * B
std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b, int n) {
  // 1. Allocate host memory for result
  std::vector<float> c(n * n);
  size_t sizeBytes = n * n * sizeof(float);

  // 2. Initialize cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // 3. Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, sizeBytes);
  cudaMalloc((void **)&d_b, sizeBytes);
  cudaMalloc((void **)&d_c, sizeBytes);

  // 4. Copy data to device
  cublasSetVector(n * n, sizeof(float), a.data(), 1, d_a, 1);
  cublasSetVector(n * n, sizeof(float), b.data(), 1, d_b, 1);

  // 5. Perform Matrix Multiplication
  // Trick for Row-Major computation using Column-Major cuBLAS:
  // C = A * B (Row Major) loop
  // C^T = B^T * A^T (Column Major)
  //
  // Since our Row-Major array 'A' looks like A^T in Column-Major,
  // and 'B' looks like B^T in Column-Major,
  // We compute C^T = B(seen as col-major) * A(seen as col-major).

  // So we invoke Sgemm:
  // Op(A) = N (No Transpose)
  // Op(B) = N (No Transpose)
  // A_arg = d_b
  // B_arg = d_a
  // C_arg = d_c

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, // m, n, k
              &alpha, d_b, n,                            // A arg (our B), lda
              d_a, n,                                    // B arg (our A), ldb
              &beta, d_c, n);                            // C arg, ldc

  // 6. Copy result back
  cublasGetVector(n * n, sizeof(float), d_c, 1, c.data(), 1);

  // 7. Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);

  return c;
}
