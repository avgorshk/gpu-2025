#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "naive_gemm_cuda.h"

namespace {
constexpr std::size_t kBlockSize = 32;

__global__ void NaiveGEMM_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                                 std::size_t n) {
  const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    float sum = 0.f;

    const float4* p_a4 = reinterpret_cast<const float4*>(a) + (row * (n / 4));
    for (int k = 0; k < n; k += 4) {
      const float4 a4 = *(p_a4++);
      sum += a4.x * b[((k + 0) * n) + col];
      sum += a4.y * b[((k + 1) * n) + col];
      sum += a4.z * b[((k + 2) * n) + col];
      sum += a4.w * b[((k + 3) * n) + col];
    }

    c[(row * n) + col] = sum;
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

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
  const std::size_t nn = n * n;
  //
  const dim3 blocks{kBlockSize, kBlockSize};
  const dim3 grid{(n + blocks.x - 1) / blocks.x, (n + blocks.y - 1) / blocks.y};

  float* devbuf;
  cudaMalloc(&devbuf, 3 * nn * sizeof(*devbuf));
  //
  float* dev_a = devbuf + (0 * nn);
  float* dev_b = devbuf + (1 * nn);
  float* dev_c = devbuf + (2 * nn);
  //
  cudaMemcpy(dev_a, a.data(), nn * sizeof(*dev_a), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b.data(), nn * sizeof(*dev_b), cudaMemcpyHostToDevice);

  NaiveGEMM_kernel<<<grid, blocks>>>(dev_a, dev_b, dev_c, n);

  std::vector<float> c;
  ResizeUninitialized(c, nn);
  //
  cudaMemcpy(c.data(), dev_c, nn * sizeof(*dev_c), cudaMemcpyDeviceToHost);

  cudaFree(devbuf);

  return c;
}
