#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(
  const std::vector<float>& a,
  const std::vector<float>& b,
  int n
) {
  std::vector<float> c(n * n, 0);

#pragma omp parallel for
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t k = 0; k < n; k++) {
      float row_elem = a[i * n + k];
      for (std::size_t j = 0; j < n; j++) {
        c[i * n + j] += row_elem * b[k * n + j];
      }
    }
  }

  return c;
}
