#ifndef CUDA_MATRIX_MULTIPLY_H
#define CUDA_MATRIX_MULTIPLY_H

#include <vector>

std::vector<float> CudaMatrixMultiply(
    const std::vector<float>& matrix_A,
    const std::vector<float>& matrix_B,
    int matrix_dim);

#endif // CUDA_MATRIX_MULTIPLY_H