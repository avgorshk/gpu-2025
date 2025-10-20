#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n) {
    std::vector<float> bT(n * n);
    std::vector<float> c(n * n);

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bT[j * n + i] = b[i * n + j];
        }
    }

    const int batch_size = 64;
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += batch_size) {
        int i1 = std::min(i0 + batch_size, n);
        for (int j0 = 0; j0 < n; j0 += batch_size) {
            int j1 = std::min(j0 + batch_size, n);
            for (int k0 = 0; k0 < n; k0 += batch_size) {
                int k1 = std::min(k0 + batch_size, n);

                for (int i = i0; i < i1; ++i) {
                    const float *a_row = &a[i * n];
                    for (int j = j0; j < j1; ++j) {
                        const float *bt_row = &bT[j * n];
                        float sum = c[i * n + j];
                        for (int k = k0; k < k1; ++k) {
                            sum += a_row[k] * bt_row[k];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
    return c;
}
