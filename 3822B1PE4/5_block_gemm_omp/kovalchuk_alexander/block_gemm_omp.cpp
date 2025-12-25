#include "block_gemm_omp.h"
#include <algorithm>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {
  std::vector<float> c(n * n, 0.0f);
  const int BLOCK_SIZE = 64;

#pragma omp parallel for schedule(dynamic) collapse(2)
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
        int i_end = std::min(ii + BLOCK_SIZE, n);
        int j_end = std::min(jj + BLOCK_SIZE, n);
        int k_end = std::min(kk + BLOCK_SIZE, n);

        for (int i = ii; i < i_end; ++i) {
          for (int k = kk; k < k_end; ++k) {
            float a_ik = a[i * n + k];
            int j = jj;

            for (; j <= j_end - 4; j += 4) {
              c[i * n + j] += a_ik * b[k * n + j];
              c[i * n + j + 1] += a_ik * b[k * n + j + 1];
              c[i * n + j + 2] += a_ik * b[k * n + j + 2];
              c[i * n + j + 3] += a_ik * b[k * n + j + 3];
            }

            for (; j < j_end; ++j) {
              c[i * n + j] += a_ik * b[k * n + j];
            }
          }
        }
      }
    }
  }

  return c;
}
