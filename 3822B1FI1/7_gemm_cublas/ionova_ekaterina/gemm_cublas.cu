// Copyright (c) 2025 Ionova-Ekaterina
#include <cstdlib>
#include <iostream>

#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gemm_cublas.h"

#define CHECK_CUDA_ERROR(callable)                                        \
  {                                                                       \
    auto codeError = callable;                                            \
    if (codeError != cudaSuccess) {                                       \
      std::cerr << "\033[1;31merror\033[0m: ";                            \
      std::cerr << cudaGetErrorString(codeError) << '\n';                 \
      std::cerr << "code error: " << static_cast<int>(codeError) << '\n'; \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n";       \
      std::exit(codeError);                                               \
    }                                                                     \
  }

#define CHECK_CUBLAS_STATUS(callable)                               \
  {                                                                 \
    auto status = callable;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                          \
      std::cerr << "\033[1;31mcublas status failed:\033[0m: ";      \
      std::cerr << static_cast<int>(status) << '\n';                \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n"; \
      std::exit(status);                                            \
    }                                                               \
  }

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b, int size) {
  cudaDeviceProp deviceProp{};
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));

  size_t countElem = size * size;
  if (a.size() != countElem || b.size() != countElem) return {};

  std::vector<float> cHost(countElem);
  auto countBytes = countElem * sizeof(float);
  float alpha = 1.0f;
  float beta = 0.0f;

  float* aDev = nullptr;
  float* bDev = nullptr;
  float* cDev = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&aDev), countBytes));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&bDev), countBytes));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&cDev), countBytes));

  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(aDev),
                              reinterpret_cast<const void*>(a.data()),
                              countBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(bDev),
                              reinterpret_cast<const void*>(b.data()),
                              countBytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle{};
  CHECK_CUBLAS_STATUS(cublasCreate(&handle));
  CHECK_CUBLAS_STATUS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size,
                                  size, &alpha, bDev, size, aDev, size, &beta,
                                  cDev, size));
  CHECK_CUBLAS_STATUS(cublasDestroy(handle));

  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(cHost.data()),
                              reinterpret_cast<void*>(cDev), countBytes,
                              cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(aDev)));
  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(bDev)));
  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(cDev)));

  return cHost;
}