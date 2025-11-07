#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>

constexpr int BLOCK_SIZE = 64;

static void multiplyBlocks(const float* restrict A_block,
                           const float* restrict B_block,
                           float* restrict C_block,
                           int block_size) {
    const int unroll_factor = 4;

    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; j += unroll_factor) {
            float sum[unroll_factor] = {0.0f};

            for (int k = 0; k < block_size; ++k) {
                float a_val = A_block[i * block_size + k];

                sum[0] += a_val * B_block[k * block_size + j + 0];
                if (j + 1 < block_size)
                    sum[1] += a_val * B_block[k * block_size + j + 1];
                if (j + 2 < block_size)
                    sum[2] += a_val * B_block[k * block_size + j + 2];
                if (j + 3 < block_size)
                    sum[3] += a_val * B_block[k * block_size + j + 3];
            }

            C_block[i * block_size + j + 0] += sum[0];
            if (j + 1 < block_size)
                C_block[i * block_size + j + 1] += sum[1];
            if (j + 2 < block_size)
                C_block[i * block_size + j + 2] += sum[2];
            if (j + 3 < block_size)
                C_block[i * block_size + j + 3] += sum[3];
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    const int num_blocks = n / BLOCK_SIZE;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int block_i = 0; block_i < num_blocks; ++block_i) {
        for (int block_j = 0; block_j < num_blocks; ++block_j) {

            float* C_block = &c[(block_i * BLOCK_SIZE) * n + (block_j * BLOCK_SIZE)];

            for (int block_k = 0; block_k < num_blocks; ++block_k) {
                const float* A_block = &a[(block_i * BLOCK_SIZE) * n + (block_k * BLOCK_SIZE)];
                const float* B_block = &b[(block_k * BLOCK_SIZE) * n + (block_j * BLOCK_SIZE)];

                multiplyBlocks(A_block, B_block, C_block, BLOCK_SIZE);
            }
        }
    }

    return c;
}