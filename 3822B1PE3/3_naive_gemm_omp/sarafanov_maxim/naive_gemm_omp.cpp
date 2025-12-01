#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n)
{
    std::vector<float> c(n * n, 0.0f);

    // Transpose B for cache-friendly access: bT[j*n + k] = b[k*n + j]
    std::vector<float> bT(n * n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            bT[j*n + i] = b[i*n + j];

    // GEMM
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const float* arow = &a[i * n];
        float* crow = &c[i * n];

        for (int j = 0; j < n; j++) {
            const float* brow = &bT[j * n];

            float sum = 0.0f;

            // Unrolling by 4 (good balance for float)
            int k = 0;
            for (; k <= n - 4; k += 4) {
                sum += arow[k]     * brow[k];
                sum += arow[k + 1] * brow[k + 1];
                sum += arow[k + 2] * brow[k + 2];
                sum += arow[k + 3] * brow[k + 3];
            }

            for (; k < n; k++) {
                sum += arow[k] * brow[k];
            }

            crow[j] = sum;
        }
    }

    return c;
}
