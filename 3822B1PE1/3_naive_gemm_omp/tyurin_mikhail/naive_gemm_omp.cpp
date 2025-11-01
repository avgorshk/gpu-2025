#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    std::vector<float> bT(n * n);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bT[j * n + i] = b[i * n + j];
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const float* a_row = &a[i * n];
        float* c_row = &c[i * n];
        for (int j = 0; j < n; ++j) {
            const float* b_row = &bT[j * n];
            float sum = 0.0f;

            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < n; ++k) {
                sum += a_row[k] * b_row[k];
            }

            c_row[j] = sum;
        }
    }

    return c;
}
