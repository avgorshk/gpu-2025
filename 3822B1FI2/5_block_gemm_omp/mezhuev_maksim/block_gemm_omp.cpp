#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

static constexpr int BLOCK_SIZE = 32;

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(static_cast<size_t>(n) * n, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < n; bk += BLOCK_SIZE) {

                int i_max = std::min(bi + BLOCK_SIZE, n);
                int j_max = std::min(bj + BLOCK_SIZE, n);
                int k_max = std::min(bk + BLOCK_SIZE, n);

                for (int i = bi; i < i_max; ++i) {
                    int c_row_offset = i * n;
                    for (int k = bk; k < k_max; ++k) {
                        float a_ik = a[static_cast<size_t>(i) * n + k];
                        int b_row_offset = k * n;
                        for (int j = bj; j < j_max; ++j) {
                            c[c_row_offset + j] +=
                                a_ik * b[b_row_offset + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
