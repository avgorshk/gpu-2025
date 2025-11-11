#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> c(n * n, 0.0f);

    std::vector<float> bT(n * n);
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            bT[j * n + i] = b[i * n + j];

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;

            int k = 0;
            int unroll_factor = 4;
            for (; k <= n - unroll_factor; k += unroll_factor) {
                sum += a[i * n + k] * bT[j * n + k];
                sum += a[i * n + k + 1] * bT[j * n + k + 1];
                sum += a[i * n + k + 2] * bT[j * n + k + 2];
                sum += a[i * n + k + 3] * bT[j * n + k + 3];
            }
            for (; k < n; ++k) {
                sum += a[i * n + k] * bT[j * n + k];
            }

            c[i * n + j] = sum;
        }
    }

    return c;
}