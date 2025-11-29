#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>


__global__ void geluKernel(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float x = input[idx];

    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;

    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);

    float exp_2x = expf(2.0f * inner);
    float tanh_val = (exp_2x - 1.0f) / (exp_2x + 1.0f);

    output[idx] = 0.5f * x * (1.0f + tanh_val);
  }
}

std::vector<float> GeluCUDA(const std::vector<float> &input) {
  int n = input.size();
  if (n == 0) {
    return std::vector<float>();
  }

  std::vector<float> output(n);

  float *d_input = nullptr;
  float *d_output = nullptr;
  size_t bytes = n * sizeof(float);

  cudaMalloc(&d_input, bytes);
  cudaMalloc(&d_output, bytes);

  cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

  cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);

  return output;
}
