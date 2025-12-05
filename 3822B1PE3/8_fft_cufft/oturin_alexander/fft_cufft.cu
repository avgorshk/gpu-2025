#include <cufft.h>

#include "fft_cufft.h"

__global__ void normalize(cufftComplex* data, int size, float n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    data[i].x *= n;
    data[i].y *= n;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
  std::vector<float> output(input.size());
  cufftHandle plan;
  cufftComplex* data;
  int pairs = input.size() / 2;
  int n = input.size() / (batch * 2);

  cudaMalloc(&data, sizeof(cufftComplex) * n * batch);
  cudaMemcpy(data, input.data(), pairs * sizeof(cufftComplex),
             cudaMemcpyHostToDevice);

  cufftPlan1d(&plan, n, CUFFT_C2C, batch);
  cufftExecC2C(plan, data, data, CUFFT_FORWARD);
  cufftExecC2C(plan, data, data, CUFFT_INVERSE);

  int block_size = 128;
  int grid_size = (input.size() + block_size - 1) / block_size;
  normalize<<<grid_size, block_size>>>(data, pairs, 1.0f / n);

  cudaMemcpy(output.data(), data, pairs * sizeof(cufftComplex),
             cudaMemcpyDeviceToHost);

  cufftDestroy(plan);
  cudaFree(data);

  return output;
}
