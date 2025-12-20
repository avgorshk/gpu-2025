#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <cstring>
#include <cmath>

constexpr int BLOCK_SIZE = 64;

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    const size_t matrix_size = n * n;
    if (a.size() != matrix_size || b.size() != matrix_size) {
        return std::vector<float>();
    }

    std::vector<float> c(matrix_size, 0.0f);

    std::vector<float> b_transposed(matrix_size);

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b_transposed[j * n + i] = b[i * n + j];
        }
    }

    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int block_i = 0; block_i < num_blocks; ++block_i) {
        for (int block_j = 0; block_j < num_blocks; ++block_j) {
            const int i_start = block_i * BLOCK_SIZE;
            const int i_end = std::min(i_start + BLOCK_SIZE, n);
            const int j_start = block_j * BLOCK_SIZE;
            const int j_end = std::min(j_start + BLOCK_SIZE, n);

            const int block_height = i_end - i_start;
            const int block_width = j_end - j_start;
            float* block_c = new float[block_height * block_width]();

            for (int block_k = 0; block_k < num_blocks; ++block_k) {
                const int k_start = block_k * BLOCK_SIZE;
                const int k_end = std::min(k_start + BLOCK_SIZE, n);
                const int block_depth = k_end - k_start;

                float* block_a = new float[block_height * block_depth];
                float* block_b = new float[block_depth * block_width];

#pragma omp simd
                for (int i = 0; i < block_height; ++i) {
                    const int src_row = i_start + i;
#pragma omp simd
                    for (int k = 0; k < block_depth; ++k) {
                        const int src_col = k_start + k;
                        block_a[i * block_depth + k] = a[src_row * n + src_col];
                    }
                }
#pragma omp simd
                for (int k = 0; k < block_depth; ++k) {
                    const int src_row = k_start + k;
#pragma omp simd
                    for (int j = 0; j < block_width; ++j) {
                        const int src_col = j_start + j;
                        block_b[k * block_width + j] = b_transposed[src_col * n + src_row];
                    }
                }

                for (int i = 0; i < block_height; ++i) {
                    const float* a_row = &block_a[i * block_depth];

                    for (int j = 0; j < block_width; j += 4) {
                        float sum0 = block_c[i * block_width + j];
                        float sum1 = block_c[i * block_width + j + 1];
                        float sum2 = block_c[i * block_width + j + 2];
                        float sum3 = block_c[i * block_width + j + 3];

                        const float* b_col0 = &block_b[j];
                        const float* b_col1 = &block_b[j + 1];
                        const float* b_col2 = &block_b[j + 2];
                        const float* b_col3 = &block_b[j + 3];

                        int k = 0;
                        for (; k <= block_depth - 4; k += 4) {
                            float a0 = a_row[k];
                            float a1 = a_row[k + 1];
                            float a2 = a_row[k + 2];
                            float a3 = a_row[k + 3];

                            sum0 += a0 * b_col0[k * block_width];
                            sum0 += a1 * b_col0[(k + 1) * block_width];
                            sum0 += a2 * b_col0[(k + 2) * block_width];
                            sum0 += a3 * b_col0[(k + 3) * block_width];

                            sum1 += a0 * b_col1[k * block_width];
                            sum1 += a1 * b_col1[(k + 1) * block_width];
                            sum1 += a2 * b_col1[(k + 2) * block_width];
                            sum1 += a3 * b_col1[(k + 3) * block_width];

                            sum2 += a0 * b_col2[k * block_width];
                            sum2 += a1 * b_col2[(k + 1) * block_width];
                            sum2 += a2 * b_col2[(k + 2) * block_width];
                            sum2 += a3 * b_col2[(k + 3) * block_width];

                            sum3 += a0 * b_col3[k * block_width];
                            sum3 += a1 * b_col3[(k + 1) * block_width];
                            sum3 += a2 * b_col3[(k + 2) * block_width];
                            sum3 += a3 * b_col3[(k + 3) * block_width];
                        }

                        for (; k < block_depth; ++k) {
                            float a_val = a_row[k];
                            sum0 += a_val * b_col0[k * block_width];
                            sum1 += a_val * b_col1[k * block_width];
                            sum2 += a_val * b_col2[k * block_width];
                            sum3 += a_val * b_col3[k * block_width];
                        }

                        block_c[i * block_width + j] = sum0;
                        block_c[i * block_width + j + 1] = sum1;
                        block_c[i * block_width + j + 2] = sum2;
                        block_c[i * block_width + j + 3] = sum3;
                    }

                    for (int j = block_width - (block_width % 4); j < block_width; ++j) {
                        float sum = block_c[i * block_width + j];
                        const float* b_col = &block_b[j];

                        for (int k = 0; k < block_depth; ++k) {
                            sum += a_row[k] * b_col[k * block_width];
                        }

                        block_c[i * block_width + j] = sum;
                    }
                }

                delete[] block_a;
                delete[] block_b;
            }

#pragma omp simd collapse(2)
            for (int i = 0; i < block_height; ++i) {
                for (int j = 0; j < block_width; ++j) {
                    c[(i_start + i) * n + (j_start + j)] = block_c[i * block_width + j];
                }
            }

            delete[] block_c;
        }
    }

    return c;
}