#include "block_gemm_omp.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>


#define BLOCK_SIZE 64

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {
  std::vector<float> c(n * n, 0.0f);

  if (a.size() != n * n || b.size() != n * n) {
    std::cerr << "Ошибка: неверный размер входных матриц" << std::endl;
    return c;
  }

  int num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
        int i_end = std::min(ii + BLOCK_SIZE, n);
        int j_end = std::min(jj + BLOCK_SIZE, n);
        int k_end = std::min(kk + BLOCK_SIZE, n);

        for (int i = ii; i < i_end; ++i) {
          int i_offset = i * n;

          for (int k = kk; k < k_end; ++k) {
            float a_ik = a[i_offset + k];
            int k_offset = k * n;

#pragma omp simd
            for (int j = jj; j < j_end; ++j) {
              c[i_offset + j] += a_ik * b[k_offset + j];
            }
          }
        }
      }
    }
  }

  return c;
}