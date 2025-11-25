#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

struct Data
{
  const float alphaMultiplier {1.0f};
  const float bettaMultiplier {0.0f};
  int size {0};
};

void multiplicator(const float* first, 
                   const float* second,
                   float* result,
                   Data& data)
{
   cublasHandle_t handle;
   cublasCreate(&handle);

   int size = data.size;
   cublasSgemm(handle, 
               CUBLAS_OP_N, 
               CUBLAS_OP_N, 
               size,
               size, 
               size,
               &data.alphaMultiplier, 
               second, 
               size, 
               first, 
               size, 
               &data.bettaMultiplier, 
               result, 
               size);
   cublasDestroy(handle);
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) 
{
  Data data;
  data.size = n;

  std::vector<float> result(n*n);
  int bytesCount = n * n * static_cast<int>(sizeof(float));

  float* firstMatrix;
  float* secondMatrix;
  float* resultMatrix;

  cudaMalloc(&firstMatrix, bytesCount);
  cudaMalloc(&secondMatrix, bytesCount);
  cudaMalloc(&resultMatrix, bytesCount);

  cudaMemcpy(firstMatrix, a.data(), bytesCount, cudaMemcpyHostToDevice);
  cudaMemcpy(secondMatrix, b.data(), bytesCount, cudaMemcpyHostToDevice);

  multiplicator(firstMatrix, secondMatrix, resultMatrix, data);

  cudaMemcpy(result.data(), resultMatrix, bytesCount, cudaMemcpyDeviceToHost);

  cudaFree(firstMatrix);
  cudaFree(secondMatrix);
  cudaFree(resultMatrix);

  return result;
}
