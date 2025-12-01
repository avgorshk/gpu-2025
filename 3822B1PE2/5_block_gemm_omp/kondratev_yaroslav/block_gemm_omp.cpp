#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    constexpr int BLOCK_SIZE = 64;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
                int i_end = std::min(i0 + BLOCK_SIZE, n);
                int j_end = std::min(j0 + BLOCK_SIZE, n);
                int k_end = std::min(k0 + BLOCK_SIZE, n);
                
                for (int i = i0; i < i_end; ++i) {
                    for (int k = k0; k < k_end; ++k) {
                        float a_ik = a[i * n + k];
                        
                        #pragma omp simd
                        for (int j = j0; j < j_end; ++j) {
                            c[i * n + j] += a_ik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
    
    return c;
}