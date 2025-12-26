// Copyright (c) 2025 Mironov-Ilya
#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gelu_cuda.h"

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

__global__ void gelu_kernel(float *y, const float *x, size_t countElem) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  /**
   * PI = 3.14159
   * constOne = 2 * sqrt(2 / PI)
   * constTwo = constOne * 0.044715
   * tmp = x * (constOne + x * x * constTwo)
   * result = x - x / (1 + exp(tmp))
   */

  if (i < countElem) {
    constexpr float constOne = 1.595769122f;
    constexpr float constTwo = constOne * 0.044715f;
    float val = x[i];
    float tmp = val * (constOne + val * val * constTwo);
    y[i] = val - val / (1.0f + __expf(tmp));
  }
}

std::vector<float> GeluCUDA(const std::vector<float> &input) {
  cudaDeviceProp deviceProp;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));

  if (input.empty()) return {};

  size_t size = input.size();
  size_t countBytes = size * sizeof(float);
  std::vector<float> output(size);
  auto threadsPerBlock = deviceProp.maxThreadsPerBlock;
  auto numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  float *inpDev = nullptr;
  float *outDev = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&inpDev), countBytes));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&outDev), countBytes));

  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(inpDev),
                              reinterpret_cast<const void *>(input.data()),
                              countBytes, cudaMemcpyHostToDevice));

  gelu_kernel<<<numBlocks, threadsPerBlock>>>(outDev, inpDev, size);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(cudaGetLastError());

  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(output.data()),
                              reinterpret_cast<void *>(outDev), countBytes,
                              cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void *>(inpDev)));
  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void *>(outDev)));

  return output;
}