#include "block_gemm_omp.h"

#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif


static inline int pick_block_size(int n) {

  if (n < 256) return 32;
  return 64;
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n) {
  const size_t nn = static_cast<size_t>(n) * n;
  std::vector<float> c(nn, 0.0f);


  std::vector<float> Bt(nn);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    const int baseBi = i * n;
    for (int j = 0; j < n; ++j) {
      Bt[j * n + i] = b[baseBi + j]; 
    }
  }

  const int BS = pick_block_size(n);


#pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < n; ii += BS) {
    for (int jj = 0; jj < n; jj += BS) {

      for (int i = ii; i < std::min(ii + BS, n); ++i) {
        const int aRowBase = i * n;
        for (int j = jj; j < std::min(jj + BS, n); ++j) {
          float sum = 0.0f;


          for (int kk = 0; kk < n; kk += BS) {
            const int kend = std::min(kk + BS, n);


            for (int k = kk; k < kend; ++k) {
              sum += a[aRowBase + k] * Bt[j * n + k];
            }
          }

          c[aRowBase + j] = sum;
        }
      }
    }
  }

  return c;
}
