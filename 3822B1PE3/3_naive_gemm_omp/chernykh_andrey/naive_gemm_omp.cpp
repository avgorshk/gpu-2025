#include "naive_gemm_omp.h"
#include <vector>
#include <omp.h>


std::vector<float> NaiveGemmOMP(
    const std::vector<float> &a,
    const std::vector<float> &b,
    int n
) {
    if (n <= 0) {
        return {};
    }

    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const float *a_row_ptr = &a[i * n];
        float *c_row_ptr = &c[i * n];
        for (int k = 0; k < n; k++) {
            float a_item = a_row_ptr[k];
            const float *b_row_ptr = &b[k * n];
#pragma omp simd
            for (int j = 0; j < n; j++) {
                c_row_ptr[j] += a_item * b_row_ptr[j];
            }
        }
    }

    return c;
}
