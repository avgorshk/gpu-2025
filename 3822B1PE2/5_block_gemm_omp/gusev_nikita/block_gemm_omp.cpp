#include "block_gemm_omp.h"
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    const int block_size = 32;
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                int i_end = (i + block_size < n) ? i + block_size : n;
                int j_end = (j + block_size < n) ? j + block_size : n;
                int k_end = (k + block_size < n) ? k + block_size : n;
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int kk = k; kk < k_end; ++kk) {
                        float a_ik = a[ii * n + kk];
                        #pragma omp simd
                        for (int jj = j; jj < j_end; ++jj) {
                            c[ii * n + jj] += a_ik * b[kk * n + jj];
                        }
                    }
                }
            }
        }
    }
    
    return c;
}

