#include "block_gemm_omp.h"

#include <omp.h>
#include <vector>
std::vector<float> BlockGemmOMP(const std::vector<float>& A,
                                const std::vector<float>& B, int n) {
  std::vector<float> C(n * n, 0.0f);
  int block_size = 32;
  #pragma omp parallel for collapse(2) schedule(guided)
  for (int i = 0; i < n; i += block_size) {
    for (int j = 0; j < n; j += block_size) {
      int i_end = std::min(i + block_size, n);
      int j_end = std::min(j + block_size, n);
      for (int k = 0; k < n; k += block_size) {
        int k_end = std::min(k + block_size, n);

        for (int i_block = i; i_block < i_end; ++i_block) {
          for (int k_block = k; k_block < k_end; ++k_block) {
            float a_ik = A[i_block * n + k_block];
            #pragma omp simd
            for (int j_block = j; j_block < j_end; ++j_block) {
              C[i_block * n + j_block] += a_ik * B[k_block * n + j_block];
            }
          }
        }
      }
    }
  }

  return C;
}
