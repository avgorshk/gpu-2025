#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

void BlockGemmKernel(const float* a_block_start, 
                     const float* b_block_start, 
                     float* c_block_start, 
                     int n, 
                     int bs) {

    
    for (int i = 0; i < bs; ++i) {
        for (int k = 0; k < bs; ++k) {
            float a_val = a_block_start[i * n + k];
            
            for (int j = 0; j < bs; ++j) {
                c_block_start[i * n + j] += a_val * b_block_start[k * n + j];
            }
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    if (n == 0) {
        return {};
    }

    std::vector<float> c(n * n, 0.0f);
    
    const int block_size = 16;
    const int block_count = n / block_size;
    
    #pragma omp parallel for
    for (int I = 0; I < block_count; ++I) {
        for (int J = 0; J < block_count; ++J) {
            float* c_block_start = &c[I * block_size * n + J * block_size];

            for (int K = 0; K < block_count; ++K) {
                const float* a_block_start = &a[I * block_size * n + K * block_size];
                const float* b_block_start = &b[K * block_size * n + J * block_size];
                
                BlockGemmKernel(a_block_start, b_block_start, c_block_start, n, block_size);
            }
        }
    }

    return c;
}