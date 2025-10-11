#include "block_gemm_omp.h"
#include <vector>
#include <cstring>
#include <cmath>

constexpr int BLOCK_SIZE = 64;

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> c(n * n, 0.0f);

    std::vector<float> b_transposed(n * n);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b_transposed[j * n + i] = b[i * n + j];
        }
    }

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int block_i = 0; block_i < num_blocks; block_i++) {
        for (int block_j = 0; block_j < num_blocks; block_j++) {

            int i_start = block_i * BLOCK_SIZE;
            int i_end = std::min(i_start + BLOCK_SIZE, n);
            int j_start = block_j * BLOCK_SIZE;
            int j_end = std::min(j_start + BLOCK_SIZE, n);

            for (int block_k = 0; block_k < num_blocks; block_k++) {
                int k_start = block_k * BLOCK_SIZE;
                int k_end = std::min(k_start + BLOCK_SIZE, n);

                for (int i = i_start; i < i_end; i++) {
                    const float* a_row = &a[i * n];
                    float* c_row = &c[i * n];

                    for (int j = j_start; j < j_end; j++) {
                        const float* b_col = &b_transposed[j * n];
                        float sum = 0.0f;

                        int k = k_start;
                        for (; k <= k_end - 4; k += 4) {
                            sum += a_row[k] * b_col[k] +
                                a_row[k + 1] * b_col[k + 1] +
                                a_row[k + 2] * b_col[k + 2] +
                                a_row[k + 3] * b_col[k + 3];
                        }
                        for (; k < k_end; k++) {
                            sum += a_row[k] * b_col[k];
                        }

                        c_row[j] += sum;
                    }
                }
            }
        }
    }

    return c;
}