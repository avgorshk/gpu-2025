#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n*n, 0);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const float* a_row = a.data() + i*n;
        float* c_row = c.data() + i*n;
        
        for (int k = 0; k < n; k++) {
            float a_ik = a_row[k];
            const float* b_row =  b.data() + k*n;
            
            #pragma omp simd
            for (int j = 0; j < n; j++) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }

    return c;
}
