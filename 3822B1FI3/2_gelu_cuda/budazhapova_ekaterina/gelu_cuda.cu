#include "gelu_cuda.h"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define M_PI 3.14159265358979323846f

__global__ void gelu_kernel(const float* input, float* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    float x = input[idx];
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    float exp_val = __expf(2.0f * inner);
    output[idx] = 0.5f * x * (1.0f + (exp_val - 1.0f) / (exp_val + 1.0f));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  int size = static_cast<int>(input.size());

  float* d_input = nullptr;
  float* d_output = nullptr;

  cudaMalloc(&d_input, size * sizeof(float));
  cudaMalloc(&d_output, size * sizeof(float));

  cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  gelu_kernel << <gridSize, blockSize >> > (d_input, d_output, size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
  }

  cudaDeviceSynchronize();

  std::vector<float> result(size);
  cudaMemcpy(result.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);

  return result;
}