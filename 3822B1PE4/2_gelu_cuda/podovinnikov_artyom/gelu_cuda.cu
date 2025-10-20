#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include "gelu_cuda.h"

__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float x = input[i];
  const float x3 = x * x * x;


  const float m = -2.0f * 0.7978845608028654f;
  const float t = x + 0.044715f * x3;

  
  out[i] = x / (1.0f + __expf(m * t));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  const int n = static_cast<int>(input.size());
  std::vector<float> res(n);
  if (n == 0) return res;

  float *d_in = nullptr, *d_out = nullptr;

  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  constexpr int BLOCK = 256;
  const int GRID = (n + BLOCK - 1) / BLOCK;

  gelu_kernel<<<GRID, BLOCK>>>(d_in, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
  return res;
}