#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int i_n = i * n;

        for (int k = 0; k < n; ++k) {
            float a_ik = a[i_n + k];
            int k_n = k * n;

            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                c[i_n + j] += a_ik * b[k_n + j];
            }
        }
    }

    return c;
}
