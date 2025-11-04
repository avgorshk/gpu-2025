#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(static_cast<size_t>(n) * n, 0.0f);

    const int block_size = 32;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            float block_c[32][32] = {0.0f};
            
            int i_end = std::min(ii + block_size, n);
            int j_end = std::min(jj + block_size, n);

            for (int kk = 0; kk < n; kk += block_size) {
                int k_end = std::min(kk + block_size, n);
                
                for (int i = ii; i < i_end; ++i) {
                    int local_i = i - ii;
                    for (int k = kk; k < k_end; ++k) {
                        const float a_ik = a[i * n + k];
                        // Vectorize inner j loop
                        #pragma omp simd
                        for (int j = jj; j < j_end; ++j) {
                            int local_j = j - jj;
                            block_c[local_i][local_j] += a_ik * b[k * n + j];
                        }
                    }
                }
            }
            
            for (int i = ii; i < i_end; ++i) {
                int local_i = i - ii;
                #pragma omp simd
                for (int j = jj; j < j_end; ++j) {
                    int local_j = j - jj;
                    c[i * n + j] = block_c[local_i][local_j];
                }
            }
        }
    }

    return c;
}
