#include "naive_gemm_omp.h"
#include <omp.h>
#include <algorithm>
#include <cstddef>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0 || a.size() != static_cast<size_t>(n * n) || b.size() != static_cast<size_t>(n * n))
        return {};

    std::vector<float> c(n * n, 0.0f);
    std::vector<float> bT(n * n);
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            bT[j * n + i] = b[i * n + j];

    const int UNROLL = 4;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            int k = 0;

#pragma omp simd reduction(+:sum)
            for (; k <= n - UNROLL; k += UNROLL) {
                sum += a[i * n + k + 0] * bT[j * n + k + 0];
                sum += a[i * n + k + 1] * bT[j * n + k + 1];
                sum += a[i * n + k + 2] * bT[j * n + k + 2];
                sum += a[i * n + k + 3] * bT[j * n + k + 3];
            }

            for (; k < n; ++k)
                sum += a[i * n + k] * bT[j * n + k];

            c[i * n + j] = sum;
        }
    }

    return c;
}
