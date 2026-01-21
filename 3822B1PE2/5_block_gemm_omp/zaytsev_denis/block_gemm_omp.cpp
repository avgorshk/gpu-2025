#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>

constexpr int BLOCK_SIZE = 64;

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {

            int i_end = std::min(bi + BLOCK_SIZE, n);
            int j_end = std::min(bj + BLOCK_SIZE, n);
            
            for (int bk = 0; bk < n; bk += BLOCK_SIZE) {
                int k_end = std::min(bk + BLOCK_SIZE, n);
                
                for (int i = bi; i < i_end; ++i) {
                    int i_offset = i * n;
                    for (int k = bk; k < k_end; ++k) {
                        float a_ik = a[i_offset + k];
                        int k_offset = k * n;
                        
                        #pragma omp simd
                        for (int j = bj; j < j_end; ++j) {
                            c[i_offset + j] += a_ik * b[k_offset + j];
                        }
                    }
                }
            }
        }
    }
    
    return c;
}