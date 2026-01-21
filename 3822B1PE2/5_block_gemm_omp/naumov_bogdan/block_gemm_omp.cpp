#include "block_gemm_omp.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

#define BLOCK_SIZE 64
#define UNROLL_FACTOR 4
#define CACHE_LINE_SIZE 64

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    const int block_size = std::min(BLOCK_SIZE, n);
    const int num_blocks = (n + block_size - 1) / block_size;
    
    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static) nowait
        for (int block_i = 0; block_i < num_blocks; ++block_i) {
            for (int block_j = 0; block_j < num_blocks; ++block_j) {
                const int i_start = block_i * block_size;
                const int i_end = std::min(i_start + block_size, n);
                const int j_start = block_j * block_size;
                const int j_end = std::min(j_start + block_size, n);
                
                float c_block[BLOCK_SIZE][BLOCK_SIZE] = {{0.0f}};
                
                for (int block_k = 0; block_k < num_blocks; ++block_k) {
                    const int k_start = block_k * block_size;
                    const int k_end = std::min(k_start + block_size, n);
                    
                    for (int i = i_start; i < i_end; ++i) {
                        const int i_local = i - i_start;
                        
                        for (int k = k_start; k < k_end; k += UNROLL_FACTOR) {
                            float a_vals[UNROLL_FACTOR];
                            for (int uk = 0; uk < UNROLL_FACTOR && (k + uk) < k_end; ++uk) {
                                a_vals[uk] = a[i * n + (k + uk)];
                            }
                            
                            for (int j = j_start; j < j_end; ++j) {
                                const int j_local = j - j_start;
                                float sum = c_block[i_local][j_local];
                                
                                for (int uk = 0; uk < UNROLL_FACTOR && (k + uk) < k_end; ++uk) {
                                    sum += a_vals[uk] * b[(k + uk) * n + j];
                                }
                                
                                c_block[i_local][j_local] = sum;
                            }
                        }
                    }
                }
                
                for (int i = i_start; i < i_end; ++i) {
                    const int i_local = i - i_start;
                    #pragma omp simd
                    for (int j = j_start; j < j_end; ++j) {
                        const int j_local = j - j_start;
                        c[i * n + j] = c_block[i_local][j_local];
                    }
                }
            }
        }
    }
    
    return c;
}