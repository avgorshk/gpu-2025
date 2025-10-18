#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
  std::size_t block_size = std::min(256, n);
  std::vector<float> c(n * n, 0);

#pragma omp parallel for
  for (std::size_t ii = 0; ii < n; ii += block_size) {
    for (std::size_t jj = 0; jj < n; jj += block_size) {
      for (std::size_t kk = 0; kk < n; kk += block_size) {
        const std::size_t i_end = std::min(ii + block_size, static_cast<std::size_t>(n));
        const std::size_t j_end = std::min(jj + block_size, static_cast<std::size_t>(n));
        const std::size_t k_end = std::min(kk + block_size, static_cast<std::size_t>(n));

        for (std::size_t i = ii; i < i_end; i++) {
          for (std::size_t k = kk; k < k_end; k++) {
            const float row_elem = a[i * n + k];
            for (std::size_t j = jj; j < j_end; j++) {
              c[i * n + j] += row_elem * b[k * n + j];
            }
          }
        }
      }
    }
  }
  return c;
}
