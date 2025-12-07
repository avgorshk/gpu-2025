#include "block_gemm_omp.h"
#include <vector>
#include <algorithm>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float> &left,
                                const std::vector<float> &right,
                                int matrix_dim) {
    std::vector<float> product(matrix_dim * matrix_dim, 0.0f);
    const int TILE = 64;
#pragma omp parallel for schedule(static) collapse(2)
    for (int block_i = 0; block_i < matrix_dim; block_i += TILE) {
        for (int block_j = 0; block_j < matrix_dim; block_j += TILE) {
            const int end_i = std::min(block_i + TILE, matrix_dim);
            const int end_j = std::min(block_j + TILE, matrix_dim);

            for (int block_k = 0; block_k < matrix_dim; block_k += TILE) {
                const int end_k = std::min(block_k + TILE, matrix_dim);

                for (int i = block_i; i < end_i; ++i) {
                    for (int j = block_j; j < end_j; ++j) {
                        float acc = product[i * matrix_dim + j];
                        for (int k = block_k; k < end_k; ++k) {
                            acc += left[i * matrix_dim + k] * right[k * matrix_dim + j];
                        }
                        product[i * matrix_dim + j] = acc;
                    }
                }
            }
        }
    }
    return product;
}