#include <cublas_v2.h>

#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b, int n) {
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);

  std::vector<float> c(n * n);

  float *inputA, *inputB, *output;
  size_t size = n * n * sizeof(float);

  cudaMalloc(&inputA, size);
  cudaMalloc(&inputB, size);
  cudaMalloc(&output, size);

  cudaMemcpy(inputA, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(inputB, b.data(), size, cudaMemcpyHostToDevice);

  float alpha = 1.0f;
  float beta = 0.0f;
  
  cublasSgemmEx(cublasHandle, 
    CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 
    &alpha, 
    inputB, CUDA_R_32F, n, 
    inputA, CUDA_R_32F, n, 
    &beta, 
    output, CUDA_R_32F, n);

  cudaMemcpy(c.data(), output, size, cudaMemcpyDeviceToHost);

  cudaFree(inputA);
  cudaFree(inputB);
  cudaFree(output);
  cublasDestroy(cublasHandle);

  return c;
}