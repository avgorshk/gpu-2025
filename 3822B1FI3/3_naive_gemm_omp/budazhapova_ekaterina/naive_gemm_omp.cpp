#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <cassert>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {
  assert(a.size() == static_cast<size_t>(n * n));
  assert(b.size() == static_cast<size_t>(n * n));

  std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;

      for (int k = 0; k < n; k++){
        sum += a[i * n + k] * b[k * n + j];
      }
      c[i * n + j] = sum;
    }
  }

  return c;
}