#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "gelu_cuda.h"

__constant__ float d_SQRT_2_OVER_PI = 0.7978845608028654f;
__constant__ float d_COEFF = 0.044715f;

__device__ __forceinline__ float gelu_device(float x) {
  float x2 = x * x;
  float x3 = x * x2;
  float inner = d_SQRT_2_OVER_PI * (x + d_COEFF * x3);

  float e = __expf(2.0f * inner);
  float t = (e - 1.0f) / (e + 1.0f);

  return 0.5f * x * (1.0f + t);
}

__global__ void gelu_kernel_vec4(const float* __restrict__ in,
                                 float* __restrict__ out, int n) {
  int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    int idx = base + lane;
    if (idx < n) {
      float x = in[idx];
      out[idx] = gelu_device(x);
    }
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  int n = static_cast<int>(input.size());
  std::vector<float> output(n);
  if (n == 0) {
    return output;
  }

  static float* d_in = nullptr;
  static float* d_out = nullptr;
  static int capacity = 0;

  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  if (n > capacity) {
    if (d_in) {
      cudaFree(d_in);
      cudaFree(d_out);
    }
    if (cudaMalloc(&d_in, bytes) != cudaSuccess ||
        cudaMalloc(&d_out, bytes) != cudaSuccess) {
      d_in = d_out = nullptr;
      capacity = 0;
      throw std::runtime_error("cudaMalloc failed");
    }
    capacity = n;
  }

  cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int threadsPerGrid = (n + 4 * blockSize - 1) / (4 * blockSize);
  gelu_kernel_vec4<<<threadsPerGrid, blockSize>>>(d_in, d_out, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("Kernel launch failed");
  }

  cudaDeviceSynchronize();

  cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

  return output;
}
