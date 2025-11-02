#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_cufft.h"

namespace {
__global__ void Normalize_kernel(cufftComplex* data, float k, int hn) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (__builtin_expect(idx < hn, 1)) {
    const float scale = 1.f / k;
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

template <typename Vec>
void ResizeUninitialized(Vec& v, std::size_t size) {
  struct stub {
    typename Vec::value_type v;
    stub() {}
  };
  reinterpret_cast<std::vector<stub>&>(v).resize(size);
}
}  // namespace

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
  const std::size_t n = input.size();
  const std::size_t hn = n / 2;

  const float kf = hn / batch;
  const int k = static_cast<int>(kf);

  cufftComplex* devbuf;

  cudaMalloc(&devbuf, hn * sizeof(*devbuf));
  cudaMemcpy(devbuf, input.data(), hn * sizeof(*devbuf), cudaMemcpyHostToDevice);

  cufftHandle plan;
  cufftPlan1d(&plan, k, CUFFT_C2C, batch);

  cufftExecC2C(plan, devbuf, devbuf, CUFFT_FORWARD);
  cufftExecC2C(plan, devbuf, devbuf, CUFFT_INVERSE);

  constexpr std::size_t blocks = 256;
  const std::size_t grid = (hn + blocks - 1) / blocks;
  Normalize_kernel<<<grid, blocks>>>(devbuf, kf, hn);

  std::vector<float> output;
  ResizeUninitialized(output, n);
  //
  cudaMemcpy(output.data(), devbuf, hn * sizeof(*devbuf), cudaMemcpyDeviceToHost);

  cufftDestroy(plan);
  cudaFree(devbuf);

  return output;
}