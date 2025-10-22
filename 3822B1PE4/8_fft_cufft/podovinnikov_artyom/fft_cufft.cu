#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_cufft.h"


__global__ void normalize_kernel(cufftComplex* data, int total, float inv_n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    data[i].x *= inv_n;
    data[i].y *= inv_n;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {

  const size_t total_floats = input.size();
  const int n = static_cast<int>(total_floats / (2ULL * batch));
  const int total_complex = n * batch;  
  const size_t bytes = sizeof(cufftComplex) * total_complex;


  cufftComplex* d_data = nullptr;
  cudaMalloc(&d_data, bytes);

 
  cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);


  cufftHandle plan;
  cufftPlan1d(&plan, n, CUFFT_C2C, batch);


  cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
  cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);


  const float inv_n = 1.0f / static_cast<float>(n);
  int block = 256;
  int grid = (total_complex + block - 1) / block;
  normalize_kernel<<<grid, block>>>(d_data, total_complex, inv_n);
  cudaDeviceSynchronize();


  std::vector<float> out(total_floats);
  cudaMemcpy(out.data(), d_data, bytes, cudaMemcpyDeviceToHost);


  cufftDestroy(plan);
  cudaFree(d_data);

  return out;
}
