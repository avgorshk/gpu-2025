#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {
  std::vector<float>result(n * n, 0.f);
  int size_block = 64;

  if (n < size_block)
    size_block = n;

  #pragma omp parallel for
  for (int i = 0; i < n; i += size_block) {
    for (int j = 0; j < n; j += size_block) {
      for (int k = 0; k < n; k += size_block) {
        for (int kk = k; kk < k + size_block; ++kk) {
          for (int ii = i; ii < i + size_block; ++ii) {
            for (int jj = j; jj < j + size_block; ++jj) {
              result[ii * n + jj] += a[ii * n + kk] * b[kk * n + jj];
            }
          }
        }
      }
    }
  }
  return result;
}