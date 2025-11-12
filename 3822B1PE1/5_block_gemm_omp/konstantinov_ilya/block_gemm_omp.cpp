#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {
  std::vector<float> c(n * n, 0.0f);

  const int blockSize = 64;

#pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < n; ii += blockSize) {
    for (int jj = 0; jj < n; jj += blockSize) {
      for (int kk = 0; kk < n; kk += blockSize) {
        int iMax = std::min(ii + blockSize, n);
        int jMax = std::min(jj + blockSize, n);
        int kMax = std::min(kk + blockSize, n);

        for (int i = ii; i < iMax; ++i) {
          for (int k = kk; k < kMax; ++k) {
            float aik = a[i * n + k];
            for (int j = jj; j < jMax; ++j) {
              c[i * n + j] += aik * b[k * n + j];
            }
          }
        }
      }
    }
  }

  return c;
}
