#include "naive_gemm_omp.h"

#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(static_cast<size_t>(n) * n, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int row = i * n;
        for (int k = 0; k < n; ++k) {
            float a_ik = a[row + k];
            int b_row = k * n;
            for (int j = 0; j < n; ++j) {
                c[row + j] += a_ik * b[b_row + j];
            }
        }
    }

    return c;
}
