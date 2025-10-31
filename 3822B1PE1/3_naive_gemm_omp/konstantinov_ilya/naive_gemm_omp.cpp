#include "naive_gemm_omp.h"
#include <omp.h>
#include <immintrin.h>
#include <algorithm>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {
  std::vector<float> c(n * n, 0.0f);

    std::vector<float> bT(n * n);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      bT[j * n + i] = b[i * n + j];

  
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      const float* a_row = &a[i * n];
      const float* b_col = &bT[j * n];

      
#pragma omp simd reduction(+:sum)
      for (int k = 0; k < n; ++k)
        sum += a_row[k] * b_col[k];

      c[i * n + j] = sum;
    }
  }

  return c;
}
