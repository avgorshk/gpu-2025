#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {
  std::vector<float> c(n * n, 0.0f);

  int BLOCK_SIZE = 64;
  if (n < 256) BLOCK_SIZE = 32;
  if (n < 128) BLOCK_SIZE = 16;

#pragma omp parallel for collapse(2)
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += BLOCK_SIZE) {

      float* c_local = &c[ii * n + jj];

      for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
        int i_end = std::min(ii + BLOCK_SIZE, n);
        int j_end = std::min(jj + BLOCK_SIZE, n);
        int k_end = std::min(kk + BLOCK_SIZE, n);

        for (int i = ii; i < i_end; i++) {
          float* c_row = &c[i * n + jj];
          const float* a_row = &a[i * n + kk];

          for (int k = kk; k < k_end; k++) {
            float aik = a_row[k - kk];
            const float* b_row = &b[k * n + jj];
            int j = jj;
            for (; j + 3 < j_end; j += 4) {
              c_row[j - jj] += aik * b_row[j - jj];
              c_row[j - jj + 1] += aik * b_row[j - jj + 1];
              c_row[j - jj + 2] += aik * b_row[j - jj + 2];
              c_row[j - jj + 3] += aik * b_row[j - jj + 3];
            }
            for (; j < j_end; j++) {
              c_row[j - jj] += aik * b_row[j - jj];
            }
          }
        }
      }
    }
  }

  return c;
}