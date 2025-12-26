#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float> &matA,
                                 const std::vector<float> &matB,
                                 int dim);

#endif // __NAIVE_GEMM_CUDA_H