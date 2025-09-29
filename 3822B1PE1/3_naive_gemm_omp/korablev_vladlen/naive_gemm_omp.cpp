//
// Created by korablev-vm on 28.09.2025.
//

#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n)) {
        return {};
        }
    std::vector answer(n * n, 0.0f);
    #pragma omp parallel for schedule(static) default(none) shared(a, b, answer, n)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const float a_ik = a[i * n + k];
            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                answer[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    return answer;
}
