#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    const float* pa = a.data();
    const float* pb = b.data();
    float* pc = c.data();

    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float aik = pa[i * n + k];
            if (aik == 0.0f) continue;
            float* row_c = pc + i * n;
            const float* row_b = pb + k * n;
            for (int j = 0; j < n; ++j) {
                row_c[j] += aik * row_b[j];
            }
        }
    }
    return c;
}
