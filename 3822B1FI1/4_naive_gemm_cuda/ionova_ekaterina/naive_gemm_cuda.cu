// Copyright (c) 2025 Ionova-Ekaterina
#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "naive_gemm_cuda.h"

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

/**
 *
 * VERSION FOR RECTANGLE MATRIXS
 *
 * __global__ void naive_gemm_kernel(float *c, const float *a, const float *b,
 *                                     const size_t m, const size_t k,
 *                                     const size_t n) {
 *   size_t iIdx = blockIdx.y * blockDim.y + threadIdx.y;
 *   size_t jIdx = blockIdx.x * blockDim.x + threadIdx.x;
 *
 *   if (iIdx < m && jIdx < n) {
 *     float resCell{};
 *     for (size_t i = 0; i < k; ++i)
 *       resCell += a[iIdx * k + i] * b[n * i + jIdx];
 *     c[iIdx * n + jIdx] = resCell;
 *   }
 * }
 *
 */

constexpr auto BLOCK_SIZE = 32;

__global__ void naive_gemm_kernel(float *c, const float *a, const float *b,
                                  const size_t size) {
  size_t iIdx = blockIdx.y * blockDim.y + threadIdx.y;
  size_t jIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (iIdx < size && jIdx < size) {
    float resCell{};
    for (size_t i = 0; i < size; ++i)
      resCell += a[iIdx * size + i] * b[size * i + jIdx];
    c[iIdx * size + jIdx] = resCell;
  }
}

static constexpr int cdiv(int a, int b) noexcept { return (a + b - 1) / b; }

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b, int size) {
  cudaDeviceProp deviceProp{};
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));

  size_t countElem = size * size;
  if (a.size() != countElem || b.size() != countElem) return {};

  std::vector<float> cHost(countElem);
  auto countBytes = countElem * sizeof(float);
  constexpr auto sizeAxis = BLOCK_SIZE;
  dim3 threadsPerBlock(sizeAxis, sizeAxis);
  dim3 numBlocks(cdiv(size, sizeAxis), cdiv(size, sizeAxis));

  float *aDev = nullptr;
  float *bDev = nullptr;
  float *cDev = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&aDev), countBytes));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&bDev), countBytes));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&cDev), countBytes));

  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(aDev),
                              reinterpret_cast<const void *>(a.data()),
                              countBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(bDev),
                              reinterpret_cast<const void *>(b.data()),
                              countBytes, cudaMemcpyHostToDevice));

  naive_gemm_kernel<<<numBlocks, threadsPerBlock>>>(cDev, aDev, bDev, size);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(cudaGetLastError());

  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(cHost.data()),
                              reinterpret_cast<void *>(cDev), countBytes,
                              cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void *>(aDev)));
  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void *>(bDev)));
  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void *>(cDev)));

  return cHost;
}