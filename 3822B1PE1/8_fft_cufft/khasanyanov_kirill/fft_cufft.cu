#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void normalizeKernel(cufftComplex *data, int size, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch) {
  if (input.empty() || batch == 0) {
    return std::vector<float>();
  }

  int n = input.size() / (2 * batch);

  std::vector<float> output(input.size());

  cufftComplex *d_data = nullptr;
  size_t bytes = n * batch * sizeof(cufftComplex);

  cudaMalloc(&d_data, bytes);

  cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

  cufftHandle plan;
  cufftPlan1d(&plan, n, CUFFT_C2C, batch);

  cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

  cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

  int totalSize = n * batch;
  float scale = 1.0f / n;
  int threadsPerBlock = 256;
  int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

  normalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, totalSize, scale);

  cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

  cufftDestroy(plan);
  cudaFree(d_data);

  return output;
}
