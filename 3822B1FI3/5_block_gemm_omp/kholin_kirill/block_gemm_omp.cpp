#include "block_gemm_omp.h"
#include <algorithm>
#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {
  std::vector<float> c(n * n, 0.0f);
  const int block_size = 64;

  const float *a_data = a.data();
  const float *b_data = b.data();
  float *c_data = c.data();

#pragma omp parallel for schedule(static)
  for (int ii = 0; ii < n; ii += block_size) {
    for (int jj = 0; jj < n; jj += block_size) {

      int i_end = std::min(ii + block_size, n);
      int j_end = std::min(jj + block_size, n);

      float c_block[64] = {0};

      for (int kk = 0; kk < n; kk += block_size) {
        int k_end = std::min(kk + block_size, n);

        for (int i = ii; i < i_end; i++) {
          std::fill(c_block, c_block + (j_end - jj), 0.0f);

          for (int k = kk; k < k_end; k++) {
            float a_val = a_data[i * n + k];
            const float *b_row = &b_data[k * n + jj];

            #pragma omp simd
            for (int j = 0; j < j_end - jj; j++) {
              c_block[j] += a_val * b_row[j];
            }
          }

          float *c_row = &c_data[i * n + jj];
          #pragma omp simd
          for (int j = 0; j < j_end - jj; j++) {
            c_row[j] += c_block[j];
          }
        }
      }
    }
  }
  return c;
}