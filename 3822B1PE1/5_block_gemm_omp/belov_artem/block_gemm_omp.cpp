#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    const int block_size = 64;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            
            int i_end = (n < ii + block_size) ? n : ii + block_size;
            int j_end = (n < jj + block_size) ? n : jj + block_size;
            
            for (int kk = 0; kk < n; kk += block_size) {
                int k_end = (n < kk + block_size) ? n : kk + block_size;
                
                for (int i = ii; i < i_end; ++i) {
                    int row_a = i * n;
                    int row_c = i * n;
                    
                    for (int k = kk; k < k_end; ++k) {
                        float a_val = a[row_a + k];
                        int row_b = k * n;
                        
                        #pragma omp simd
                        for (int j = jj; j < j_end; ++j) {
                            c[row_c + j] += a_val * b[row_b + j];
                        }
                    }
                }
            }
        }
    }
    
    return c;
}
