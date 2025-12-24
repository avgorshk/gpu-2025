#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
 const std::vector<float>& b,
 int n) {
 if (a.size() != n * n || b.size() != n * n) {
  throw std::invalid_argument("Matrix size mismatch");
 }

 cublasHandle_t handle;
 cublasStatus_t status = cublasCreate(&handle);
 if (status != CUBLAS_STATUS_SUCCESS) {
  throw std::runtime_error("Failed to initialize cuBLAS");
 }

 float *d_a, *d_b, *d_c;
 size_t size = n * n * sizeof(float);
 
 cudaError_t err;
 err = cudaMalloc((void**)&d_a, size);
 if (err != cudaSuccess) {
  cublasDestroy(handle);
  throw std::runtime_error("Failed to allocate device memory for matrix A");
 }
 
 err = cudaMalloc((void**)&d_b, size);
 if (err != cudaSuccess) {
  cudaFree(d_a);
  cublasDestroy(handle);
  throw std::runtime_error("Failed to allocate device memory for matrix B");
 }
 
 err = cudaMalloc((void**)&d_c, size);
 if (err != cudaSuccess) {
  cudaFree(d_a);
  cudaFree(d_b);
  cublasDestroy(handle);
  throw std::runtime_error("Failed to allocate device memory for matrix C");
 }

 err = cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
 if (err != cudaSuccess) {
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);
  throw std::runtime_error("Failed to copy matrix A to device");
 }
 
 err = cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
 if (err != cudaSuccess) {
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);
  throw std::runtime_error("Failed to copy matrix B to device");
 }

 float *h_result;
 err = cudaMallocHost((void**)&h_result, size);
 if (err != cudaSuccess) {
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);
  throw std::runtime_error("Failed to allocate pinned host memory");
 }

 const float alpha = 1.0f;
 const float beta = 0.0f;
 
 status = cublasSgemm(handle,
                      CUBLAS_OP_T,
                      CUBLAS_OP_T,
                      n,
                      n,
                      n,
                      &alpha,
                      d_b,
                      n,
                      d_a,
                      n,
                      &beta,
                      d_c,
                      n);
 
 if (status != CUBLAS_STATUS_SUCCESS) {
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFreeHost(h_result);
  cublasDestroy(handle);
  throw std::runtime_error("cuBLAS matrix multiplication failed");
 }

 err = cudaMemcpy(h_result, d_c, size, cudaMemcpyDeviceToHost);
 if (err != cudaSuccess) {
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFreeHost(h_result);
  cublasDestroy(handle);
  throw std::runtime_error("Failed to copy result from device");
 }

 cudaFree(d_a);
 cudaFree(d_b);
 cudaFree(d_c);
 cublasDestroy(handle);

 std::vector<float> result_row_major(n * n);
 for (int i = 0; i < n; i++) {
  for (int j = 0; j < n; j++) {
   result_row_major[i * n + j] = h_result[j * n + i];
  }
 }

 cudaFreeHost(h_result);
 return result_row_major;
}

