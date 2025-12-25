#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            int k = 0;
            for (; k <= n - 4; k += 4) {
                sum += a[i * n + k] * b[k * n + j] +
                       a[i * n + k + 1] * b[(k + 1) * n + j] +
                       a[i * n + k + 2] * b[(k + 2) * n + j] +
                       a[i * n + k + 3] * b[(k + 3) * n + j];
            }
            for (; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    
    return c;
}