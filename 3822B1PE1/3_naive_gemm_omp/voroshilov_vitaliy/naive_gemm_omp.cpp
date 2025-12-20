#include "naive_gemm_omp.h"

#include <vector>
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float aij = a[i * n + j];
            #pragma omp simd
            for (int k = 0; k < n; k++) {
                c[i * n + k] += aij * b[k * n + k];
            }
        }
    }

    return c;
}
