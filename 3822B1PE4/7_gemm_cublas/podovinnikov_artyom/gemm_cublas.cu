#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b, int n) {
  const size_t elems = static_cast<size_t>(n) * n;
  const size_t bytes = elems * sizeof(float);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  cudaMalloc(&dA, bytes);
  cudaMalloc(&dB, bytes);
  cudaMalloc(&dC, bytes);

  cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              n,             
              n,           
              n,             
              &alpha, dB, n, 
              dA, n,    
              &beta, dC, n
  );


  cudaDeviceSynchronize();


  std::vector<float> c(elems);
  cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost);


  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return c;
}
