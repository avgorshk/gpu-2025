#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b, int n) {
    std::vector<float> C(n * n, 0.0f);

#pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const float* aRow = &a[row * n];
            float acc = 0.0f;
            int k = 0;
            for (; k <= n - 4; k += 4) {
                acc += aRow[k] * b[(k)*n + col] + aRow[k + 1] * b[(k + 1) * n + col] +
                    aRow[k + 2] * b[(k + 2) * n + col] +
                    aRow[k + 3] * b[(k + 3) * n + col];
            }
            for (; k < n; ++k) {
                acc += aRow[k] * b[k * n + col];
            }
            C[row * n + col] = acc;
        }
    }
    return C;
}