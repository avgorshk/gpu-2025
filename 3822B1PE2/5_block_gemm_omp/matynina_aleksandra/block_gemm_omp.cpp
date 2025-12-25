#include "block_gemm_omp.h"
#include <omp.h>
#include <cstring>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
 const std::vector<float>& b,
 int n) {
    const int block_size = 64;
    const int block_count = n / block_size;
    
    std::vector<float> c(n * n, 0.0f);
    
    #pragma omp parallel for collapse(2)
    for (int block_i = 0; block_i < block_count; ++block_i) {
        for (int block_j = 0; block_j < block_count; ++block_j) {
            const int i_start = block_i * block_size;
            const int j_start = block_j * block_size;
            
            float block_c[block_size * block_size];
            std::memset(block_c, 0, sizeof(block_c));
            
            for (int block_k = 0; block_k < block_count; ++block_k) {
                const int k_start = block_k * block_size;
                
                for (int i = 0; i < block_size; ++i) {
                    const int a_row = i_start + i;
                    const int a_base = a_row * n + k_start;
                    const int c_row_idx = i * block_size;
                    
                    for (int j = 0; j < block_size; ++j) {
                        const int b_col = j_start + j;
                        float sum = 0.0f;
                        
                        #pragma omp simd reduction(+:sum)
                        for (int k = 0; k < block_size; ++k) {
                            const int b_row = k_start + k;
                            sum += a[a_base + k] * b[b_row * n + b_col];
                        }
                        
                        block_c[c_row_idx + j] += sum;
                    }
                }
            }
            
            for (int i = 0; i < block_size; ++i) {
                const int c_row = i_start + i;
                const int block_row_idx = i * block_size;
                const int c_base = c_row * n + j_start;
                
                #pragma omp simd
                for (int j = 0; j < block_size; ++j) {
                    c[c_base + j] = block_c[block_row_idx + j];
                }
            }
        }
    }
    
    return c;
}

