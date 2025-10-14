#include "block_gemm_omp.h"

#pragma GCC optimize("O3")  // i just want to see if this works
std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n) {
  std::vector<float> output(n * n, 0);
  constexpr int block_size = 32;

  if (n < block_size || n % block_size != 0) throw;  // sorry ig

#pragma omp parallel for collapse(2)
  for (int i_bl = 0; i_bl < n; i_bl += block_size) {
    for (int j_bl = 0; j_bl < n; j_bl += block_size) {
      for (int k_bl = 0; k_bl < n; k_bl += block_size) {
        for (int i = i_bl; i < i_bl + block_size; i++) {
          float* out_ptr = &output[i * n + j_bl];
          const float* a_ptr = &a[i * n + k_bl];

          for (int k = k_bl; k < k_bl + block_size; k++) {
            const float a_idxval = a_ptr[k - k_bl];
            const float* b_ptr = &b[k * n + j_bl];

            out_ptr[0] += a_idxval * b_ptr[0];
            out_ptr[1] += a_idxval * b_ptr[1];
            out_ptr[2] += a_idxval * b_ptr[2];
            out_ptr[3] += a_idxval * b_ptr[3];
            out_ptr[4] += a_idxval * b_ptr[4];
            out_ptr[5] += a_idxval * b_ptr[5];
            out_ptr[6] += a_idxval * b_ptr[6];
            out_ptr[7] += a_idxval * b_ptr[7];
            out_ptr[8] += a_idxval * b_ptr[8];
            out_ptr[9] += a_idxval * b_ptr[9];
            out_ptr[10] += a_idxval * b_ptr[10];
            out_ptr[11] += a_idxval * b_ptr[11];
            out_ptr[12] += a_idxval * b_ptr[12];
            out_ptr[13] += a_idxval * b_ptr[13];
            out_ptr[14] += a_idxval * b_ptr[14];
            out_ptr[15] += a_idxval * b_ptr[15];
            out_ptr[16] += a_idxval * b_ptr[16];
            out_ptr[17] += a_idxval * b_ptr[17];
            out_ptr[18] += a_idxval * b_ptr[18];
            out_ptr[19] += a_idxval * b_ptr[19];
            out_ptr[20] += a_idxval * b_ptr[20];
            out_ptr[21] += a_idxval * b_ptr[21];
            out_ptr[22] += a_idxval * b_ptr[22];
            out_ptr[23] += a_idxval * b_ptr[23];
            out_ptr[24] += a_idxval * b_ptr[24];
            out_ptr[25] += a_idxval * b_ptr[25];
            out_ptr[26] += a_idxval * b_ptr[26];
            out_ptr[27] += a_idxval * b_ptr[27];
            out_ptr[28] += a_idxval * b_ptr[28];
            out_ptr[29] += a_idxval * b_ptr[29];
            out_ptr[30] += a_idxval * b_ptr[30];
            out_ptr[31] += a_idxval * b_ptr[31];
            // i didnt type this by hand, ok?
          }
        }
      }
    }
  }

  return output;
}
