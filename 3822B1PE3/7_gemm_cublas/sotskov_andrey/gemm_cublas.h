#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& matrixA,
                              const std::vector<float>& matrixB,
                              int matrixSize);

#endif // __GEMM_CUBLAS_H
