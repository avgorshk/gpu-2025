#include "naive_gemm_omp.h"
#include <vector>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    
    std::vector<float> c(n * n, 0.0f);
    if (n == 0) return c;
    
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* c_ptr = c.data();
        
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int i_offset = i * n;
        
        for (int k = 0; k < n; ++k) {
            const float a_ik = a_ptr[i_offset + k];
            const int k_offset = k * n;
            
            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                c_ptr[i_offset + j] += a_ik * b_ptr[k_offset + j];
            }
        }
    }
    
    return c;
}