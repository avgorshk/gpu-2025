#include "fft_cufft.h"

__global__ void normalize_vector(float* _data, size_t _length, float _norm_factor) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < _length) {
    _data[idx] *= _norm_factor;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
  size_t size = input.size();
  std::vector<float> result(size, 0.f);
  int n = size / (2 * batch);
  size = size * sizeof(float);

  cufftComplex* data_device;
  cudaMalloc(&data_device, sizeof(cufftComplex) * n * batch);
  cudaMemcpy(data_device, input.data(), size, cudaMemcpyHostToDevice);


  cufftHandle plan;
  cufftPlan1d(&plan, n, CUFFT_C2C, batch);

  cufftExecC2C(plan, data_device, data_device, CUFFT_FORWARD);
  cufftExecC2C(plan, data_device, data_device, CUFFT_INVERSE);

  int n_th_block = 256;
  int n_bl_grid = (size + n_th_block - 1) / n_th_block;
  float norm_factor = 1.0f / static_cast<float>(n);

  normalize_vector << <n_bl_grid, n_th_block >> > (reinterpret_cast<float*>(data_device), input.size(), norm_factor);

  cudaMemcpy(result.data(), data_device, size, cudaMemcpyDeviceToHost);

  cufftDestroy(plan);
  cudaFree(data_device);

  return result;
}