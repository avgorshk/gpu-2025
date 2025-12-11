#include "naive_gemm_omp.h"
#include <vector>
#include <omp.h>
#include <algorithm>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    if (n <= 0 ||
        a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n)) {
        return std::vector<float>();
    }

    std::vector<float> result(n * n, 0.0f);
    std::vector<float> b_transposed(n * n);

    const int BLOCK_SIZE = 64;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            int i_end = std::min(i + BLOCK_SIZE, n);
            int j_end = std::min(j + BLOCK_SIZE, n);
            for (int ii = i; ii < i_end; ++ii) {
                for (int jj = j; jj < j_end; ++jj) {
                    b_transposed[jj * n + ii] = b[ii * n + jj];
                }
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const float* a_row = &a[i * n];
        float* c_row = &result[i * n];

        for (int j = 0; j < n; ++j) {
            const float* b_col = &b_transposed[j * n];
            float sum = 0.0f;

            int k = 0;
            for (; k <= n - 8; k += 8) {
                sum += a_row[k] * b_col[k];
                sum += a_row[k + 1] * b_col[k + 1];
                sum += a_row[k + 2] * b_col[k + 2];
                sum += a_row[k + 3] * b_col[k + 3];
                sum += a_row[k + 4] * b_col[k + 4];
                sum += a_row[k + 5] * b_col[k + 5];
                sum += a_row[k + 6] * b_col[k + 6];
                sum += a_row[k + 7] * b_col[k + 7];
            }

            for (; k < n; ++k) {
                sum += a_row[k] * b_col[k];
            }

            c_row[j] = sum;
        }
    }

    return result;
}