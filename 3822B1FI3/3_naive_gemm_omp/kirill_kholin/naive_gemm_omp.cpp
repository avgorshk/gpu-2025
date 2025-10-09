#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {
  std::vector<float> c(n * n, 0.0f);

  #pragma omp parallel for
  for (std::size_t i = 0; i < n; i++) {
    float *c_row = &c[i * n];
    for (std::size_t k = 0; k < n; k++) {
      float a_ik = a[i * n + k];
      const float *b_row = &b[k * n];
      #pragma omp simd
      for (std::size_t j = 0; j < n; j++) {
        c_row[j] += a_ik * b_row[j];
      }
    }
  }

  return c;
}