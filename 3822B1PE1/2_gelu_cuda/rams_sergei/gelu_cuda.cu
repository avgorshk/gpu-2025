#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define SQRT_2_OVER_PI (-2.0f * 0.7978845608028654f)

__global__ void kernel(const float* input, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = input[i];
    out[i] = x / (1.0f + std::exp(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  int n = input.size();
  std::vector<float> res(n);
  float *device_input, *device_res;

  cudaMalloc(&device_input, n * sizeof(float));
  cudaMalloc(&device_res, n * sizeof(float));

  cudaMemcpy(device_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  constexpr int block_size = 256;
  int blocks = (n + block_size - 1) / block_size;

  kernel<<<blocks, block_size>>>(device_input, device_res, n);

  cudaMemcpy(res.data(), device_res, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_input);
  cudaFree(device_res);

  return res;
}
