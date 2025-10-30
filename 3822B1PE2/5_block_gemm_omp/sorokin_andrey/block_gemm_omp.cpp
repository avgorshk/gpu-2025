#include "block_gemm_omp.h"
#include <cstring>
#include <cmath>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> result_matrix(n * n, 0.0f);
    std::vector<float> transposed_b(n * n);
    #pragma omp parallel for schedule(static)
    for (int row_idx = 0; row_idx < n; row_idx++) {
        for (int col_idx = 0; col_idx < n; col_idx++) {
            transposed_b[col_idx * n + row_idx] = b[row_idx * n + col_idx];
        }
    }
    int total_blocks_dim = (n + 64 - 1) / 64;
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int outer_block_idx = 0; outer_block_idx < total_blocks_dim; outer_block_idx++) {
        for (int inner_block_idx = 0; inner_block_idx < total_blocks_dim; inner_block_idx++) {
            int start_i = outer_block_idx * 64;
            int end_i = (start_i + 64 < n) ? start_i + 64 : n;
            int start_j = inner_block_idx * 64;
            int end_j = (start_j + 64 < n) ? start_j + 64 : n;

            for (int depth_block_idx = 0; depth_block_idx < total_blocks_dim; depth_block_idx++) {
                int start_k = depth_block_idx * 64;
                int end_k = (start_k + 64 < n) ? start_k + 64 : n;
                for (int i_pos = start_i; i_pos < end_i; i_pos++) {
                    const float* a_current_row = a.data() + i_pos * n;
                    float* c_current_row = result_matrix.data() + i_pos * n;

                    for (int j_pos = start_j; j_pos < end_j; j_pos++) {
                        const float* b_current_col = transposed_b.data() + j_pos * n;
                        float partial_sum = 0.0f;
                        int k_pos = start_k;
                        int remaining_iterations = end_k - start_k;
                        int vectorizable_iterations = remaining_iterations - (remaining_iterations % 4);

                        for (; k_pos < start_k + vectorizable_iterations; k_pos += 4) {
                            partial_sum += 
                                a_current_row[k_pos] * b_current_col[k_pos] +
                                a_current_row[k_pos + 1] * b_current_col[k_pos + 1] +
                                a_current_row[k_pos + 2] * b_current_col[k_pos + 2] +
                                a_current_row[k_pos + 3] * b_current_col[k_pos + 3];
                        }
                        for (; k_pos < end_k; k_pos++) {
                            partial_sum += a_current_row[k_pos] * b_current_col[k_pos];
                        }
                        c_current_row[j_pos] = c_current_row[j_pos] + partial_sum;
                    }
                }
            }
        }
    }

    return result_matrix;
}