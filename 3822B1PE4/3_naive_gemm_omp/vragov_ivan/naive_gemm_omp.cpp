#include "naive_gemm_omp.h"
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    // Cache-friendly: iterate over i, then k, then j
    // This improves cache locality for matrix A
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_ik = a[i * n + k];
            // Vectorize inner loop over j
            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    
    return c;
}

