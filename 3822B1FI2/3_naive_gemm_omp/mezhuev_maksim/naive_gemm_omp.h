#ifndef NAIVE_GEMM_OMP_H
#define NAIVE_GEMM_OMP_H

#include <vector>

std::vector<float> MatrixMultiplyOMP(
    const std::vector<float>& mat_a,
    const std::vector<float>& mat_b,
    int matrix_size);

#endif // NAIVE_GEMM_OMP_H