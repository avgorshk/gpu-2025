#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>
#include <cmath>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    int block_size = 32;
    if (n >= 512) block_size = 64;
    if (n >= 1024) block_size = 128;
    if (n >= 2048) block_size = 256;
    
    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static)
        for (int i_block = 0; i_block < n; i_block += block_size) {
            for (int j_block = 0; j_block < n; j_block += block_size) {
                
                for (int k_block = 0; k_block < n; k_block += block_size) {
                    int i_end = std::min(i_block + block_size, n);
                    int j_end = std::min(j_block + block_size, n);
                    int k_end = std::min(k_block + block_size, n);
                    
                    for (int i = i_block; i < i_end; ++i) {
                        for (int k = k_block; k < k_end; ++k) {
                            float a_val = a[i * n + k];
                            float* c_ptr = &c[i * n + j_block];
                            const float* b_ptr = &b[k * n + j_block];
                            
                            #pragma omp simd
                            for (int j = 0; j < j_end - j_block; ++j) {
                                c_ptr[j] += a_val * b_ptr[j];
                            }
                        }
                    }
                }
            }
        }
    }
    
    return c;
}