#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const float a_i_j = a[i * n + j];
            #pragma omp simd
            for (int k = 0; k < n; ++k) {
                c[i * n + k] += a_i_j * b[j * n + k];
            }
        }
    }

    return c;
}
