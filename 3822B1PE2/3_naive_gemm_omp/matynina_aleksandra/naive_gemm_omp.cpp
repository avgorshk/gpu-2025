#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n, 0.0f);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_ik = a[i * n + k];
            
            const float* b_row = &b[k * n];
            float* c_row = &c[i * n];
            
            int j = 0;
            for (; j < n - 3; j += 4) {
                c_row[j] += a_ik * b_row[j];
                c_row[j + 1] += a_ik * b_row[j + 1];
                c_row[j + 2] += a_ik * b_row[j + 2];
                c_row[j + 3] += a_ik * b_row[j + 3];
            }
            
            for (; j < n; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
    
    return c;
}

