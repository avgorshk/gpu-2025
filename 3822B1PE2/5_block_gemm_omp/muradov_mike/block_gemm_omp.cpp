#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>
#include <algorithm>

#define BLOCK_SIZE 64

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    
    std::vector<float> c(n * n, 0.0f);
    if (n == 0) return c;
    
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* c_ptr = c.data();
    
    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < num_blocks; ++bi) {
        for (int bj = 0; bj < num_blocks; ++bj) {
            
            const int i_start = bi * BLOCK_SIZE;
            const int i_end = std::min(i_start + BLOCK_SIZE, n);
            const int j_start = bj * BLOCK_SIZE;
            const int j_end = std::min(j_start + BLOCK_SIZE, n);
            
            for (int bk = 0; bk < num_blocks; ++bk) {
                const int k_start = bk * BLOCK_SIZE;
                const int k_end = std::min(k_start + BLOCK_SIZE, n);
                
                for (int i = i_start; i < i_end; ++i) {
                    const int i_offset = i * n;
                    
                    for (int k = k_start; k < k_end; ++k) {
                        const float a_ik = a_ptr[i_offset + k];
                        const int k_offset = k * n;
                        
                        #pragma omp simd
                        for (int j = j_start; j < j_end; ++j) {
                            c_ptr[i_offset + j] += a_ik * b_ptr[k_offset + j];
                        }
                    }
                }
            }
        }
    }
    
    return c;
}