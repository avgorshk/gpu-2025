#include "naive_gemm_omp.h"
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> result_matrix(n * n, 0.0f);
    int matrix_size = n;

#pragma omp parallel for collapse(2) schedule(static)
    for (int row_idx = 0; row_idx < matrix_size; ++row_idx) {
        for (int col_idx = 0; col_idx < matrix_size; ++col_idx) {
            float accum = 0.0f;
            for (int inner_idx = 0; inner_idx < matrix_size; ++inner_idx) {
                float a_val = a[row_idx * matrix_size + inner_idx];
                float b_val = b[inner_idx * matrix_size + col_idx];
                accum += a_val * b_val;
            }
            result_matrix[row_idx * matrix_size + col_idx] = accum;
        }
    }

    return result_matrix;
}