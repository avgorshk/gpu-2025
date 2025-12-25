#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {
  std::vector<float> res(n * n, 0.0f);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        res[i * n + j] += a[i * n + k] * b[k * n + j];
      }
    }
  }
  return res;
}