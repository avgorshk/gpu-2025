#include <cuda_runtime.h>

#include <vector>

#include "block_gemm_cuda.h"

namespace {
constexpr std::size_t kTileSize = 64;
constexpr std::size_t kSubtileSize = 4;
__global__ void BlockGEMM_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                                 std::size_t n) {
  __shared__ float shared_a[kTileSize][kTileSize];
  __shared__ float shared_b[kTileSize][kTileSize];
  //
  float res[kSubtileSize][kSubtileSize]{0};

  const std::size_t bx = blockIdx.x, by = blockIdx.y;
  const std::size_t tx = threadIdx.x, ty = threadIdx.y;

  const std::size_t row = by * kTileSize + ty * kSubtileSize;
  const std::size_t col = bx * kTileSize + tx * kSubtileSize;

  const std::size_t tiles = (n + kTileSize - 1) / kTileSize;
  for (std::size_t t = 0; t < tiles; t++) {
    for (std::size_t i = 0; i < kSubtileSize; i++) {
      const std::size_t l_row = ty * kSubtileSize + i;
      const std::size_t a_row = row + i;
      const std::size_t b_row = t * kTileSize + l_row;
      for (std::size_t j = 0; j < kSubtileSize; j++) {
        const std::size_t l_col = tx * kSubtileSize + j;
        const std::size_t a_col = t * kTileSize + l_col;
        const std::size_t b_col = col + j;

        shared_a[l_row][l_col] = (a_row < n && a_col < n) ? a[a_row * n + a_col] : 0.f;
        shared_b[l_row][l_col] = (b_row < n && b_col < n) ? b[b_row * n + b_col] : 0.f;
      }
    }

    __syncthreads();

    for (std::size_t k = 0; k < kTileSize; k++) {
      for (std::size_t i = 0; i < kSubtileSize; i++) {
#pragma unroll
        for (std::size_t j = 0; j < kSubtileSize; j++) {
          res[i][j] += shared_a[ty * kSubtileSize + i][k] * shared_b[k][tx * kSubtileSize + j];
        }
      }
    }

    __syncthreads();
  }

  for (std::size_t i = 0; i < kSubtileSize; i++) {
    const std::size_t g = row + i;
    if (g >= n) {
      break;
    }
    auto* ck = c + (g * n);
#pragma unroll
    for (std::size_t j = 0; j < kSubtileSize; j++) {
      const std::size_t l = col + j;
      if (l >= n) {
        break;
      }
      ck[l] = res[i][j];
    }
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

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
  const std::size_t nn = n * n;
  //
  const auto blocks = [&]() -> dim3 {
    const unsigned int v = kTileSize / kSubtileSize;
    return {v, v};
  }();
  const auto grid = [n]() -> dim3 {
    const unsigned int v = (n + kTileSize - 1) / kTileSize;
    return {v, v};
  }();

  float* devbuf;
  cudaMalloc(&devbuf, 3 * nn * sizeof(*devbuf));
  //
  float* dev_a = devbuf + (0 * nn);
  float* dev_b = devbuf + (1 * nn);
  float* dev_c = devbuf + (2 * nn);
  //
  cudaMemcpy(dev_a, a.data(), nn * sizeof(*dev_a), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b.data(), nn * sizeof(*dev_b), cudaMemcpyHostToDevice);

  BlockGEMM_kernel<<<grid, blocks>>>(dev_a, dev_b, dev_c, n);

  std::vector<float> c;
  ResizeUninitialized(c, nn);
  //
  cudaMemcpy(c.data(), dev_c, nn * sizeof(*dev_c), cudaMemcpyDeviceToHost);

  cudaFree(devbuf);

  return c;
}
