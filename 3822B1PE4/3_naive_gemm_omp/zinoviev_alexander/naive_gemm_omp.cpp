#include "naive_gemm_omp.h"
#include <vector>
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int row_offset = i * n;
        for (int k = 0; k < n; ++k) {
            const float a_ik = a[row_offset + k];
            const int b_row_offset = k * n;
            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                c[row_offset + j] += a_ik * b[b_row_offset + j];
            }
        }
    }
    
    return c;
}