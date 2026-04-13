#include "block_gemm_omp.h"
#include <algorithm>
#include <omp.h>
#include <vector>


constexpr int BLOCK_SIZE = 64;
constexpr int UNROLL_FACTOR = 4;

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {
  if (n <= 0) {
    return std::vector<float>();
  }

  size_t matrix_size = static_cast<size_t>(n) * n;
  std::vector<float> c(matrix_size, 0.0f);

  const float *__restrict__ a_ptr = a.data();
  const float *__restrict__ b_ptr = b.data();
  float *__restrict__ c_ptr = c.data();

  int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

#pragma omp parallel for collapse(2) schedule(static)
  for (int block_i = 0; block_i < num_blocks; ++block_i) {
    for (int block_j = 0; block_j < num_blocks; ++block_j) {
      int i_start = block_i * BLOCK_SIZE;
      int i_end = std::min(i_start + BLOCK_SIZE, n);
      int j_start = block_j * BLOCK_SIZE;
      int j_end = std::min(j_start + BLOCK_SIZE, n);

      for (int block_k = 0; block_k < num_blocks; ++block_k) {
        int k_start = block_k * BLOCK_SIZE;
        int k_end = std::min(k_start + BLOCK_SIZE, n);

        for (int i = i_start; i < i_end; ++i) {
          const int i_offset = i * n;
          const float *a_row = &a_ptr[i_offset];
          float *c_row = &c_ptr[i_offset];

          for (int k = k_start; k < k_end; ++k) {
            const float a_ik = a_row[k];
            const int k_offset = k * n;
            const float *b_row = &b_ptr[k_offset];

            int j = j_start;
            int j_unrolled = j_end - ((j_end - j_start) % UNROLL_FACTOR);

            for (; j < j_unrolled; j += UNROLL_FACTOR) {
              c_row[j] += a_ik * b_row[j];
              c_row[j + 1] += a_ik * b_row[j + 1];
              c_row[j + 2] += a_ik * b_row[j + 2];
              c_row[j + 3] += a_ik * b_row[j + 3];
            }

#pragma omp simd
            for (int jj = j; jj < j_end; ++jj) {
              c_row[jj] += a_ik * b_row[jj];
            }
          }
        }
      }
    }
  }

  return c;
}
