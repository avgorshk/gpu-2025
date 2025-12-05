#include "block_gemm_omp.h"
#include <vector>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    const int block_dim = 64;
    std::vector<float> result(n * n, 0.0f);

    int block_count = (n + block_dim - 1) / block_dim;

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < block_count; bi++) {
        for (int bj = 0; bj < block_count; bj++) {

            int i_start = bi * block_dim;
            int i_end = std::min(i_start + block_dim, n);
            int j_start = bj * block_dim;
            int j_end = std::min(j_start + block_dim, n);

            for (int bk = 0; bk < block_count; bk++) {
                int k_start = bk * block_dim;
                int k_end = std::min(k_start + block_dim, n);

                for (int i = i_start; i < i_end; i++) {
                    for (int k = k_start; k < k_end; k++) {
                        float a_val = a[i * n + k];
                        for (int j = j_start; j < j_end; j++) {
                            result[i * n + j] += a_val * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return result;
}