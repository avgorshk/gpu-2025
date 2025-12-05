#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {

    std::vector<float> c(n * n, 0.0f);

    std::vector<float> bT(n * n);

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bT[j * n + i] = b[i * n + j];
        }
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const float* a_row = &a[i * n];
            const float* b_col = &bT[j * n];

            float sum = 0.0f;

#pragma omp simd reduction(+:sum)
            for (int k = 0; k < n; ++k) {
                sum += a_row[k] * b_col[k];
            }

            c[i * n + j] = sum;
        }
    }

    return c;
}