#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <cstring>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    const size_t matrix_size = n * n;
    if (a.size() != matrix_size || b.size() != matrix_size) {
        return std::vector<float>();
    }

    std::vector<float> c(matrix_size, 0.0f);
    std::vector<float> b_transposed(matrix_size);

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b_transposed[j * n + i] = b[i * n + j];
        }
    }


#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const float* a_row = &a[i * n];
        float* c_row = &c[i * n];

        for (int j = 0; j < n; j += 4) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

            const float* b_col0 = &b_transposed[j * n];
            const float* b_col1 = &b_transposed[(j + 1) * n];
            const float* b_col2 = &b_transposed[(j + 2) * n];
            const float* b_col3 = &b_transposed[(j + 3) * n];

            int k = 0;
            for (; k <= n - 4; k += 4) {
                float a0 = a_row[k];
                float a1 = a_row[k + 1];
                float a2 = a_row[k + 2];
                float a3 = a_row[k + 3];

                sum0 += a0 * b_col0[k];
                sum0 += a1 * b_col0[k + 1];
                sum0 += a2 * b_col0[k + 2];
                sum0 += a3 * b_col0[k + 3];

                sum1 += a0 * b_col1[k];
                sum1 += a1 * b_col1[k + 1];
                sum1 += a2 * b_col1[k + 2];
                sum1 += a3 * b_col1[k + 3];

                sum2 += a0 * b_col2[k];
                sum2 += a1 * b_col2[k + 1];
                sum2 += a2 * b_col2[k + 2];
                sum2 += a3 * b_col2[k + 3];

                sum3 += a0 * b_col3[k];
                sum3 += a1 * b_col3[k + 1];
                sum3 += a2 * b_col3[k + 2];
                sum3 += a3 * b_col3[k + 3];
            }

            for (; k < n; ++k) {
                float a_val = a_row[k];
                sum0 += a_val * b_col0[k];
                sum1 += a_val * b_col1[k];
                sum2 += a_val * b_col2[k];
                sum3 += a_val * b_col3[k];
            }

            c_row[j] = sum0;
            c_row[j + 1] = sum1;
            c_row[j + 2] = sum2;
            c_row[j + 3] = sum3;
        }
    }

    return c;
}