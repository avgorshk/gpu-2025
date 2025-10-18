#include "naive_gemm_omp.h"
#include <omp.h>
#include <algorithm>
#include <cstddef>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    if (n <= 0) return {};
    if (a.size() != static_cast<size_t>(n * n) || b.size() != static_cast<size_t>(n * n)) {
        return {};
    }

    std::vector<float> c(n * n, 0.0f);

    std::vector<float> bT(n * n);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            bT[j * n + i] = b[i * n + j];

    const int blockSize = 64;

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                int i_end = std::min(ii + blockSize, n);
                int j_end = std::min(jj + blockSize, n);
                int k_end = std::min(kk + blockSize, n);

                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        float sum = c[i * n + j];
                        int k = kk;
                        // Loop unrolling by 4
                        for (; k <= k_end - 4; k += 4) {
                            sum += a[i * n + k + 0] * bT[j * n + k + 0];
                            sum += a[i * n + k + 1] * bT[j * n + k + 1];
                            sum += a[i * n + k + 2] * bT[j * n + k + 2];
                            sum += a[i * n + k + 3] * bT[j * n + k + 3];
                        }
                        for (; k < k_end; ++k)
                            sum += a[i * n + k] * bT[j * n + k];
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }

    return c;
}
