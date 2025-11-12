#include "block_gemm_omp.h"
#include <vector>
#include <algorithm>
#include <cmath>

constexpr int BLOCK_SIZE = 64;

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> c(n * n, 0.0f);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

#pragma omp parallel for collapse(2) schedule(static)
    for (int block_i = 0; block_i < num_blocks; block_i++) {
        for (int block_j = 0; block_j < num_blocks; block_j++) {

            int i_start = block_i * BLOCK_SIZE;
            int i_end = std::min(i_start + BLOCK_SIZE, n);
            int j_start = block_j * BLOCK_SIZE;
            int j_end = std::min(j_start + BLOCK_SIZE, n);

            std::vector<float> local_block(BLOCK_SIZE * BLOCK_SIZE, 0.0f);

            for (int block_k = 0; block_k < num_blocks; block_k++) {
                int k_start = block_k * BLOCK_SIZE;
                int k_end = std::min(k_start + BLOCK_SIZE, n);

                for (int i = i_start; i < i_end; i++) {
                    int local_i = i - i_start;
                    const float* a_ptr = &a[i * n + k_start];

                    for (int k = k_start; k < k_end; k++) {
                        float a_val = a_ptr[k - k_start];
                        const float* b_ptr = &b[k * n + j_start];

                        for (int j = j_start; j < j_end; j++) {
                            int local_j = j - j_start;
                            local_block[local_i * BLOCK_SIZE + local_j] += a_val * b_ptr[j - j_start];
                        }
                    }
                }
            }

            for (int i = i_start; i < i_end; i++) {
                int local_i = i - i_start;
                for (int j = j_start; j < j_end; j++) {
                    int local_j = j - j_start;
                    c[i * n + j] = local_block[local_i * BLOCK_SIZE + local_j];
                }
            }
        }
    }

    return c;
}