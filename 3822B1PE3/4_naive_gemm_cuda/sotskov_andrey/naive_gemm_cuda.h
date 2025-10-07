#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& matrix_a,
                                 const std::vector<float>& matrix_b,
                                 int matrix_size);

std::vector<float> NaiveGemmCUDA_v2(const std::vector<float>& matrix_a,
                                    const std::vector<float>& matrix_b,
                                    int matrix_size);

#endif  // __NAIVE_GEMM_CUDA_H