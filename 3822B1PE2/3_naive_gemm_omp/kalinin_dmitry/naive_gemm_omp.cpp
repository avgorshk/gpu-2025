#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    std::vector<float> b_transposed(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b_transposed[j * n + i] = b[i * n + j];
        }
    }
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            const float* a_row = &a[i * n];
            const float* b_col = &b_transposed[j * n];
            
            int k = 0;
            for (; k < n - 7; k += 8) {
                sum += a_row[k] * b_col[k];
                sum += a_row[k + 1] * b_col[k + 1];
                sum += a_row[k + 2] * b_col[k + 2];
                sum += a_row[k + 3] * b_col[k + 3];
                sum += a_row[k + 4] * b_col[k + 4];
                sum += a_row[k + 5] * b_col[k + 5];
                sum += a_row[k + 6] * b_col[k + 6];
                sum += a_row[k + 7] * b_col[k + 7];
            }
            for (; k < n; ++k) {
                sum += a_row[k] * b_col[k];
            }
            c[i * n + j] = sum;
        }
    }
    
    return c;
}

