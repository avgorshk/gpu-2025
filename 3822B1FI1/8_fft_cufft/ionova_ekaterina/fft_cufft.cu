// Copyright (c) 2025 Ionova-Ekaterina
#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"
#include "fft_cufft.h"

#define CHECK_CUDA_ERROR(callable)                                        \
  {                                                                       \
    auto codeError = callable;                                            \
    if (codeError != cudaSuccess) {                                       \
      std::cerr << "\033[1;31merror\033[0m: ";                            \
      std::cerr << cudaGetErrorString(codeError) << '\n';                 \
      std::cerr << "code error: " << static_cast<int>(codeError) << '\n'; \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n";       \
      std::exit(static_cast<int>(codeError));                             \
    }                                                                     \
  }

#define CHECK_CUFFT_RESULT(callable)                                \
  {                                                                 \
    auto result = callable;                                         \
    if (result != CUFFT_SUCCESS) {                                  \
      std::cerr << "\033[1;31mcufft result failed:\033[0m: ";       \
      std::cerr << static_cast<int>(result) << '\n';                \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n"; \
      std::exit(static_cast<int>(result));                          \
    }                                                               \
  }

__global__ void normalize_kernel(float* x, size_t size, float coef) {
  size_t i = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) +
             static_cast<size_t>(threadIdx.x);
  if (i < size) {
    x[i] *= coef;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
  cudaDeviceProp deviceProp{};
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));

  if (input.empty()) return {};

  auto size = input.size();
  auto elemPerBatch = size / batch >> 1;
  auto countBytes = sizeof(cufftComplex) * elemPerBatch * batch;
  auto threadsPerBlock = deviceProp.maxThreadsPerBlock;
  auto numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  std::vector<float> output(size);

  cufftComplex* signal = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&signal, countBytes));
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(signal),
                              reinterpret_cast<const void*>(input.data()),
                              countBytes, cudaMemcpyHostToDevice));

  cufftHandle handle{};
  CHECK_CUFFT_RESULT(cufftPlan1d(&handle, elemPerBatch, CUFFT_C2C, batch));
  CHECK_CUFFT_RESULT(cufftExecC2C(handle, signal, signal, CUFFT_FORWARD));
  CHECK_CUFFT_RESULT(cufftExecC2C(handle, signal, signal, CUFFT_INVERSE));
  CHECK_CUFFT_RESULT(cufftDestroy(handle));

  normalize_kernel<<<numBlocks, threadsPerBlock>>>(
      reinterpret_cast<float*>(signal), size, 1.0f / elemPerBatch);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(cudaGetLastError());

  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(output.data()),
                              reinterpret_cast<void*>(signal), countBytes,
                              cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(signal)));

  return output;
}