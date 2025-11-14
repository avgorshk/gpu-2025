#include "naive_gemm_omp.h"

#include <omp.h>
#include <algorithm>
#include <cstddef>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0 || a.size() != static_cast<size_t>(n) * n || b.size() != static_cast<size_t>(n) * n)
        return {};

    std::vector<float> c(static_cast<size_t>(n) * n, 0.0f);
    std::vector<float> bT(static_cast<size_t>(n) * n);

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bT[j * n + i] = b[i * n + j];
        }
    }

    const int UNROLL = 4;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;

            int limit = n - (n % UNROLL);
#pragma omp simd reduction(+:sum)
            for (int kk = 0; kk < limit; kk += UNROLL) {
                sum += a[i * n + kk + 0] * bT[j * n + kk + 0];
                sum += a[i * n + kk + 1] * bT[j * n + kk + 1];
                sum += a[i * n + kk + 2] * bT[j * n + kk + 2];
                sum += a[i * n + kk + 3] * bT[j * n + kk + 3];
            }
            for (int k = limit; k < n; ++k) {
                sum += a[i * n + k] * bT[j * n + k];
            }

            c[i * n + j] = sum;
        }
    }

    return c;
}
