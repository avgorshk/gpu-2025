#include "block_gemm_omp.h"
#include <algorithm>
#include <vector>
#include <omp.h>

std::vector<float> BlockGemmOMP(
    const std::vector<float> &a,
    const std::vector<float> &b,
    int n
) {
    if (n <= 0) {
        return {};
    }
    constexpr int BLOCK_SIZE = 64;

    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for schedule(static)
    for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < n; bk += BLOCK_SIZE) {
                int i_max = std::min(bi + BLOCK_SIZE, n);
                int j_max = std::min(bj + BLOCK_SIZE, n);
                int k_max = std::min(bk + BLOCK_SIZE, n);

                for (int i = bi; i < i_max; ++i) {
                    const float *a_row_ptr = &a[i * n];
                    float *c_row_ptr = &c[i * n];

                    for (int k = bk; k < k_max; ++k) {
                        float a_item = a_row_ptr[k];
                        const float *b_row_ptr = &b[k * n];

#pragma omp simd
                        for (int j = bj; j < j_max; ++j) {
                            c_row_ptr[j] += a_item * b_row_ptr[j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
