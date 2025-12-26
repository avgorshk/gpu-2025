#include "block_gemm_omp.h"
#include <algorithm>
#include <omp.h>
#include <vector>


std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {
  std::vector<float> c(n * n, 0.0f);

  // Choose block size (L1/L2 cache friendly)
  // 64 floats * 64 floats * 4 bytes = 16KB
  const int BS = 64;

  // Pointer optimization: Get raw pointers to avoid vector indirection in inner
  // loops
  const float *A_ptr = a.data();
  const float *B_ptr = b.data();
  float *C_ptr = c.data();

// Parallelize over result blocks (BlockI, BlockJ)
#pragma omp parallel for collapse(2)
  for (int bi = 0; bi < n; bi += BS) {
    for (int bj = 0; bj < n; bj += BS) {

      // Local buffer for accumulator to reduce false sharing and global writes
      // Static size array is faster than vector allocation
      float local_c[BS * BS] = {0.0f};

      // Effective block size (handle edges if n is not multiple of BS,
      // though task says n is power of 2, so it likely fits perfectly if BS=64)
      int iMax = std::min(bi + BS, n);
      int jMax = std::min(bj + BS, n);
      int iLen = iMax - bi;
      int jLen = jMax - bj;

      // Loop over K-blocks
      for (int bk = 0; bk < n; bk += BS) {
        int kMax = std::min(bk + BS, n);

        // --- Block multiplication: local_c += A_sub * B_sub ---

        // Optimized innermost loops
        // Iterate i, k, j for vectorization on j
        for (int i = 0; i < iLen; ++i) {

          // Pre-calculate pointer to A row
          // Global index for A: (bi + i, bk)
          const float *A_row = A_ptr + (bi + i) * n;

          // Pre-calculate pointer to Local C row
          float *local_C_row = local_c + i * BS;

          for (int k = bk; k < kMax; ++k) {
            float aik = A_row[k];

            // Pointer to B row
            // Global index for B: (k, bj)
            const float *B_row = B_ptr + k * n + bj;

// Vectorize accumulation
#pragma omp simd
            for (int j = 0; j < jLen; ++j) {
              local_C_row[j] += aik * B_row[j];
            }
          }
        }
      }

      // Write local result back to global memory C
      for (int i = 0; i < iLen; ++i) {
        float *C_row = C_ptr + (bi + i) * n + bj;
        float *local_C_row = local_c + i * BS;

#pragma omp simd
        for (int j = 0; j < jLen; ++j) {
          C_row[j] = local_C_row[j];
        }
      }
    }
  }

  return c;
}
