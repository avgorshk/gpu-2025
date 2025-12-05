#include <omp.h>

#include <algorithm>
#include <vector>

#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n) {
  const int N = n;
  const std::size_t NN = static_cast<std::size_t>(N) * N;

  std::vector<float> c(NN, 0.0f);
  if (N == 0) {
    return c;
  }

  std::vector<float> Bt(NN);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; ++i) {
    const float* b_row = b.data() + static_cast<std::size_t>(i) * N;
    float* bt_col_base = Bt.data() + i;
    for (int j = 0; j < N; ++j) {
      bt_col_base[static_cast<std::size_t>(j) * N] = b_row[j];
    }
  }

  const int BLOCK = 64;

#pragma omp parallel for collapse(2) schedule(static)
  for (int bi = 0; bi < N; bi += BLOCK) {
    for (int bj = 0; bj < N; bj += BLOCK) {
      const int i_end = std::min(bi + BLOCK, N);
      const int j_end = std::min(bj + BLOCK, N);

      for (int bk = 0; bk < N; bk += BLOCK) {
        const int k_end = std::min(bk + BLOCK, N);

        for (int i = bi; i < i_end; ++i) {
          const float* a_row = a.data() + static_cast<std::size_t>(i) * N;
          float* c_row = c.data() + static_cast<std::size_t>(i) * N;

          for (int j = bj; j < j_end; ++j) {
            const float* b_col = Bt.data() + static_cast<std::size_t>(j) * N;

            float sum = c_row[j];

            int k = bk;
            int limit = k_end - ((k_end - bk) & 3);

            for (; k < limit; k += 4) {
              sum += a_row[k] * b_col[k] + a_row[k + 1] * b_col[k + 1] +
                     a_row[k + 2] * b_col[k + 2] + a_row[k + 3] * b_col[k + 3];
            }
            for (; k < k_end; ++k) {
              sum += a_row[k] * b_col[k];
            }

            c_row[j] = sum;
          }
        }
      }
    }
  }

  return c;
}
