#include "naive_gemm_omp.h"

#include <omp.h>

#include <iostream>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n) {
  std::vector<float> c(n * n, 0.0f);
  #pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      #pragma omp simd reduction(+ : sum)
      for (size_t k = 0; k < n; ++k) {
        sum += a[i * n + k] * b[k * n + j];
      }

      c[i * n + j] = sum;
    }
  }

  return c;
}