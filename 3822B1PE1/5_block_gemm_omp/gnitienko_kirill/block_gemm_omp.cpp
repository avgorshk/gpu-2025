#include "block_gemm_omp.h"
#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c(n * n, 0.0f);

    int block_size;
    if (n <= 64) {
        block_size = 8;
    }
    else if (n <= 256) {
        block_size = 16;
    }
    else if (n <= 1024) {
        block_size = 32;
    }
    else {
        block_size = 64;
    }

    int num_blocks = n / block_size;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i_block = 0; i_block < num_blocks; ++i_block) {
        for (int j_block = 0; j_block < num_blocks; ++j_block) {

            int i_start = i_block * block_size;
            int j_start = j_block * block_size;

            for (int k_block = 0; k_block < num_blocks; ++k_block) {
                int k_start = k_block * block_size;

                for (int i = 0; i < block_size; ++i) {
                    int a_base = (i_start + i) * n + k_start;

                    for (int k = 0; k < block_size; ++k) {
                        int b_base = (k_start + k) * n + j_start;
#pragma omp simd
                        for (int j = 0; j < block_size; ++j) {
                            int c_idx = (i_start + i) * n + j_start + j;
                            c[c_idx] += a[a_base + k] * b[b_base + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}