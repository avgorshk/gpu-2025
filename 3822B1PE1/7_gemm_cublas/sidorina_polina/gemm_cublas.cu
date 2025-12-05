#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

struct GemmParams
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int matrix_size = 0;
};

void ComputeGemm(const float* matrix_a, 
                 const float* matrix_b,
                 float* matrix_c,
                 GemmParams& params)
{
   cublasHandle_t handle;
   cublasCreate(&handle);

   int n = params.matrix_size;
   cublasSgemm(handle, 
               CUBLAS_OP_N, 
               CUBLAS_OP_N, 
               n,
               n, 
               n,
               &params.alpha, 
               matrix_b, 
               n, 
               matrix_a, 
               n, 
               &params.beta, 
               matrix_c, 
               n);
   cublasDestroy(handle);
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) 
{
  GemmParams params;
  params.matrix_size = n;

  std::vector<float> result(n * n);
  int byte_count = n * n * static_cast<int>(sizeof(float));

  float* d_matrix_a;
  float* d_matrix_b;
  float* d_matrix_c;

  cudaMalloc(&d_matrix_a, byte_count);
  cudaMalloc(&d_matrix_b, byte_count);
  cudaMalloc(&d_matrix_c, byte_count);

  cudaMemcpy(d_matrix_a, a.data(), byte_count, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix_b, b.data(), byte_count, cudaMemcpyHostToDevice);

  ComputeGemm(d_matrix_a, d_matrix_b, d_matrix_c, params);

  cudaMemcpy(result.data(), d_matrix_c, byte_count, cudaMemcpyDeviceToHost);

  cudaFree(d_matrix_a);
  cudaFree(d_matrix_b);
  cudaFree(d_matrix_c);

  return result;
}