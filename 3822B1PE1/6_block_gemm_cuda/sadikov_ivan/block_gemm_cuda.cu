#include "block_gemm_cuda.h"
#include <iostream>
#include <cuda_runtime.h>

namespace constants
{
  constexpr int threadsCount{32};
}

__global__ void kernel(const float* first, const float* second, float* result, size_t n) 
{
  auto x = blockIdx.x * constants::threadsCount + threadIdx.x;
  auto y = blockIdx.y * constants::threadsCount + threadIdx.y;
  if (x >= n || y >= n) 
  {
    return;
  }
  
  __shared__ float firstShared[constants::threadsCount][constants::threadsCount];
  __shared__ float secondShared[constants::threadsCount][constants::threadsCount];

  for (size_t block = 0; block < (n + constants::threadsCount - 1) / constants::threadsCount; ++block)
  {
    firstShared[threadIdx.y][threadIdx.x] = first[y * n + (block * constants::threadsCount + threadIdx.x)];
    secondShared[threadIdx.y][threadIdx.x] = second[x + n * (block * constants::threadsCount + threadIdx.y)];
    __syncthreads();
    
    auto* currentFirstShared = firstShared[threadIdx.y];
    auto& currentResult = result[y * n + x];
    for (int k = 0; k < constants::threadsCount; ++k) 
    {
      currentResult += currentFirstShared[k] * secondShared[k][threadIdx.x];
    }
    __syncthreads();
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) 
{
  std::vector<float> result(n * n);
  size_t size = sizeof(float) * n * n;
  
  float* first;
  float* second;
  float* cudaResult;
  
  cudaMalloc(&first, size);
  cudaMalloc(&second, size);
  cudaMalloc(&cudaResult, size);

  cudaMemcpy(first, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(second, b.data(), size, cudaMemcpyHostToDevice);

  dim3 threads(constants::threadsCount, constants::threadsCount);
  int blocksSize = (n + constants::threadsCount - 1) / constants::threadsCount;
  dim3 blocksCount(blocksSize, blocksSize);

  kernel<<<blocksCount, threads>>>(first, second, cudaResult, n);

  cudaMemcpy(result.data(), cudaResult, size, cudaMemcpyDeviceToHost);

  cudaFree(first);
  cudaFree(second);
  cudaFree(cudaResult);

  return result;
}