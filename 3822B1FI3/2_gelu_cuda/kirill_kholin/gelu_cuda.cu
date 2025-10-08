#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

constexpr float sqrt_coeff = 0.7978845608f;
constexpr float half_coeff = 0.5f;
constexpr float two_coeff = 2.0f;
constexpr float one_coeff = 1.0f;
constexpr float special_coeff = 0.044715f;
constexpr float negative_two_coeff = -2.0f;

__global__ void kernel(const float *input, float *out, int n) {
  int grid_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (grid_index < n) {
    float x = input[grid_index];
    float x_cube = x * x * x;
    float arg_tanh = sqrt_coeff * (x + special_coeff * x_cube);
    float exp_func =
        two_coeff / (one_coeff + expf(negative_two_coeff * arg_tanh));
    out[grid_index] = half_coeff * x * exp_func;
  }
}

std::vector<float> GeluCUDA(const std::vector<float> &input) {
  int n = input.size();
  std::vector<float> output(n);
  float *dev_input;
  float *dev_output;

  cudaMalloc(&dev_input, n * sizeof(float));
  cudaMalloc(&dev_output, n * sizeof(float));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMemcpyAsync(dev_input, input.data(), n * sizeof(float),
                  cudaMemcpyHostToDevice, stream);

  constexpr int block_size = 256;
  int blocks = (n + block_size - 1) / block_size;

  kernel<<<blocks, block_size, 0, stream>>>(dev_input, dev_output, n);

  cudaMemcpyAsync(output.data(), dev_output, n * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaFree(dev_input);
  cudaFree(dev_output);

  return output;
}