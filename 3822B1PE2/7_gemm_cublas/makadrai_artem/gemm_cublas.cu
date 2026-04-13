#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

static cublasHandle_t handle = nullptr;
static float *d_a = nullptr;
static float *d_b = nullptr;
static float *d_c = nullptr;
static cudaStream_t stream = nullptr;
static size_t allocated_size = 0;
static bool initialized = false;

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b, int n) {
  if (n <= 0) {
    return std::vector<float>();
  }

  size_t matrix_size = static_cast<size_t>(n) * n;
  size_t bytes = matrix_size * sizeof(float);

  if (!initialized) {
    cublasCreate(&handle);
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    initialized = true;
  }

  if (d_a == nullptr || allocated_size < matrix_size) {
    if (d_a != nullptr) {
      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
    }
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    allocated_size = matrix_size;
  }

  cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_b, n, d_a, n,
              &beta, d_c, n);

  std::vector<float> c(matrix_size);

  cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  return c;
}
