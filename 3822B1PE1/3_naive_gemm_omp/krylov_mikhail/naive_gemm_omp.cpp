#include "naive_gemm_omp.h"

#include <omp.h>

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

std::vector<float> NaiveGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
  std::vector<float> c;
  ResizeUninitialized(c, n * n);

  const float* __restrict__ p_a = a.data();
  const float* __restrict__ p_b = b.data();
  float* __restrict__ p_c = c.data();

#pragma omp parallel for simd schedule(static)
  for (std::size_t i = 0; i < n; i++) {
#pragma omp collapse(2) simd
    for (std::size_t k = 0; k < n; k++) {
      const float x{p_a[i * n + k]};
#pragma omp unroll(8) simd
      for (std::size_t j = 0; j < n; j++) {
        p_c[i * n + j] += x * p_b[k * n + j];
      }
    }
  }

  return c;
}