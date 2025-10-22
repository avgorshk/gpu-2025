#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_cufft.h"

// Нормализация на устройстве: делим каждую комплексную точку на n
__global__ void normalize_kernel(cufftComplex* data, int total, float inv_n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    data[i].x *= inv_n;
    data[i].y *= inv_n;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
  // input.size() = 2 * n * batch  → восстановим n
  const size_t total_floats = input.size();
  const int n = static_cast<int>(total_floats / (2ULL * batch));
  const int total_complex = n * batch;  // сколько комплексных чисел всего
  const size_t bytes = sizeof(cufftComplex) * total_complex;

  // Выделение памяти на девайсе (in-place буфер)
  cufftComplex* d_data = nullptr;
  cudaMalloc(&d_data, bytes);

  // Хост → девайс (в том же интерливинге float2)
  cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

  // План: батч из batch сигналов длины n (C2C, один буфер)
  cufftHandle plan;
  cufftPlan1d(&plan, n, CUFFT_C2C, batch);

  // Прямое и обратное преобразования (in-place)
  cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
  cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

  // Нормализация на устройстве: делим на n
  const float inv_n = 1.0f / static_cast<float>(n);
  int block = 256;
  int grid = (total_complex + block - 1) / block;
  normalize_kernel<<<grid, block>>>(d_data, total_complex, inv_n);
  cudaDeviceSynchronize();

  // Девайс → хост
  std::vector<float> out(total_floats);
  cudaMemcpy(out.data(), d_data, bytes, cudaMemcpyDeviceToHost);

  // Очистка
  cufftDestroy(plan);
  cudaFree(d_data);

  return out;
}
