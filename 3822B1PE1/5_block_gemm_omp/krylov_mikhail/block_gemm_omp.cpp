#include "block_gemm_omp.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
template <typename Vec>
void ResizeUninitialized(Vec& v, std::size_t size) {
  struct stub {
    typename Vec::value_type v;
    stub() {}
  };
  reinterpret_cast<std::vector<stub>&>(v).resize(size);
}
}  // namespace

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n_) {
  const std::size_t n{static_cast<std::size_t>(n_)};

  std::vector<float> c;
  ResizeUninitialized(c, n * n);

  const float* __restrict__ p_a = a.data();
  const float* __restrict__ p_b = b.data();
  float* __restrict__ p_c = c.data();

  const std::size_t block_size = std::min(256ul, n);

#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t ii = 0; ii < n; ii += block_size) {
    for (std::size_t jj = 0; jj < n; jj += block_size) {
      for (std::size_t kk = 0; kk < n; kk += block_size) {
        const std::size_t ir = std::min(ii + block_size, n);
        const std::size_t jr = std::min(jj + block_size, n);
        const std::size_t kr = std::min(kk + block_size, n);

        for (std::size_t i = ii; i < ir; ++i) {
          for (std::size_t k = kk; k < kr; ++k) {
            const float m{p_a[i * n + k]};
            const float* __restrict__ p_bx = p_b + (k * n) + jj;
            float* __restrict__ p_cx = p_c + (i * n) + jj;

            const std::size_t dj{jr - jj};

#pragma omp unroll(8) simd
            for (std::size_t j = 0; j < dj; ++j) {
              p_cx[j] += m * p_bx[j];
            }
          }
        }
      }
    }
  }

  return c;
}