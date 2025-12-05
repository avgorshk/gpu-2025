#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> MatrixMultiplyOMP(
    const std::vector<float>& mat_a,
    const std::vector<float>& mat_b,
    int matrix_size) {
    
    std::vector<float> mat_c(matrix_size * matrix_size, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < matrix_size; ++row) {
        for (int col = 0; col < matrix_size; ++col) {
            float dot_product = 0.0f;
            for (int k = 0; k < matrix_size; ++k) {
                dot_product += mat_a[row * matrix_size + k] * 
                               mat_b[k * matrix_size + col];
            }
            mat_c[row * matrix_size + col] = dot_product;
        }
    }

    return mat_c;
}