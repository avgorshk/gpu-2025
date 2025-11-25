#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    const int block_size = 64;
    
    #pragma omp parallel for schedule(static) collapse(2)
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                
                const int i_max = std::min(ii + block_size, n);
                const int j_max = std::min(jj + block_size, n);
                const int k_max = std::min(kk + block_size, n);
                
                for (int i = ii; i < i_max; ++i) {
                    const int row_offset = i * n;
                    
                    for (int k = kk; k < k_max; ++k) {
                        const float a_ik = a[row_offset + k];
                        const int b_row_offset = k * n;
                        
                        #pragma omp simd
                        for (int j = jj; j < j_max; ++j) {
                            c[row_offset + j] += a_ik * b[b_row_offset + j];
                        }
                    }
                }
            }
        }
    }
    
    return c;
}