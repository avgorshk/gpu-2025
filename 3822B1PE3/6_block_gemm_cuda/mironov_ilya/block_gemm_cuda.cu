// Copyright (c) 2025 Mironov-Ilya
#include <cstdlib>
#include <iostream>

#include "block_gemm_cuda.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
 * __global__ void matmul_block_kernel(float *c, const float *a, const float *b,
 *                                     const size_t mSize, const size_t kSize,
 *                                     const size_t nSize) {
 *   size_t iGlob = blockIdx.y * BLOCK_SIZE + threadIdx.y;
 *   size_t jGlob = blockIdx.x * BLOCK_SIZE + threadIdx.x;
 *   size_t iLoc = threadIdx.y;
 *   size_t jLoc = threadIdx.x;
 *
 *   __shared__ float aShared[BLOCK_SIZE * BLOCK_SIZE];
 *   __shared__ float bShared[BLOCK_SIZE * BLOCK_SIZE];
 *   int numBlocks = (kSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
 *   float resCell{};
 *
 *   for (int i = 0; i < numBlocks; ++i) {
 *     if (iGlob < mSize && i * BLOCK_SIZE + jLoc < kSize) {
 *       aShared[iLoc * BLOCK_SIZE + jLoc] =
 *           a[iGlob * kSize + i * BLOCK_SIZE + jLoc];
 *     } else {
 *       aShared[iLoc * BLOCK_SIZE + jLoc] = 0.0f;
 *     }
 *
 *     if (jGlob < nSize && i * BLOCK_SIZE + iLoc < kSize) {
 *       bShared[iLoc * BLOCK_SIZE + jLoc] =
 *           b[i * BLOCK_SIZE * nSize + iLoc * nSize + jGlob];
 *     } else {
 *       bShared[iLoc * BLOCK_SIZE + jLoc] = 0.0f;
 *     }
 *
 *     __syncthreads();
 *     for (int j = 0; j < BLOCK_SIZE; ++j) {
 *       resCell +=
 *           aShared[iLoc * BLOCK_SIZE + j] * bShared[j * BLOCK_SIZE + jLoc];
 *     }
 *     __syncthreads();
 *   }
 *
 *   if (iGlob < mSize && jGlob < nSize) {
 *     c[iGlob * nSize + jGlob] = resCell;
 *   }
 * }
 *
 */

constexpr auto BLOCK_SIZE = 32;

__global__ void block_gemm_kernel(float *c, const float *a, const float *b,
                                  const size_t size) {
  size_t iGlob = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  size_t jGlob = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  size_t iLoc = threadIdx.y;
  size_t jLoc = threadIdx.x;

  __shared__ float aShared[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float bShared[BLOCK_SIZE * BLOCK_SIZE];
  int numBlocks = gridDim.x;
  float resCell{};

  for (int i = 0; i < numBlocks; ++i) {
    aShared[iLoc * BLOCK_SIZE + jLoc] = a[iGlob * size + i * BLOCK_SIZE + jLoc];
    bShared[iLoc * BLOCK_SIZE + jLoc] =
        b[i * BLOCK_SIZE * size + iLoc * size + jGlob];

    __syncthreads();
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      resCell +=
          aShared[iLoc * BLOCK_SIZE + j] * bShared[j * BLOCK_SIZE + jLoc];
    }
    __syncthreads();
  }

  if (iGlob < size && jGlob < size) {
    c[iGlob * size + jGlob] = resCell;
  }
}

static constexpr int cdiv(int a, int b) noexcept { return (a + b - 1) / b; }

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
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

  block_gemm_kernel<<<numBlocks, threadsPerBlock>>>(cDev, aDev, bDev, size);
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