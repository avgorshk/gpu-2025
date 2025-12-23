#include "block_gemm_omp.h"
#include <cstring>
#include <omp.h>
#include <vector>


std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {
  std::vector<float> c(n * n, 0.0f);

  const int block_size = 64;

#pragma omp parallel for collapse(2) default(none) shared(a, b, c, n)          \
    firstprivate(block_size) schedule(static)
  for (int I = 0; I < n; I += block_size) {
    for (int J = 0; J < n; J += block_size) {
      for (int K = 0; K < n; K += block_size) {
        int i_end = (I + block_size > n) ? n : I + block_size;
        int j_end = (J + block_size > n) ? n : J + block_size;
        int k_end = (K + block_size > n) ? n : K + block_size;

        for (int i = I; i < i_end; ++i) {
          for (int j = J; j < j_end; ++j) {
            float sum = 0.0f;
            for (int k = K; k < k_end; ++k) {
              sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] += sum;
          }
        }
      }
    }
  }

  return c;
}