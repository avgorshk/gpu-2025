#include "naive_gemm_omp.h"
#include <cmath>
#include <omp.h>
#include <vector>


std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {
  std::vector<float> c(n * n, 0.0f);

  if (a.size() != n * n || b.size() != n * n) {
    return c;
  }

  std::vector<float> b_transposed(n * n);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      b_transposed[j * n + i] = b[i * n + j];
    }
  }

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      int idx_a = i * n;
      int idx_b = j * n;

      int k = 0;
      for (; k <= n - 4; k += 4) {
        sum += a[idx_a + k] * b_transposed[idx_b + k] +
               a[idx_a + k + 1] * b_transposed[idx_b + k + 1] +
               a[idx_a + k + 2] * b_transposed[idx_b + k + 2] +
               a[idx_a + k + 3] * b_transposed[idx_b + k + 3];
      }

      for (; k < n; ++k) {
        sum += a[idx_a + k] * b_transposed[idx_b + k];
      }

      c[i * n + j] = sum;
    }
  }

  return c;
}