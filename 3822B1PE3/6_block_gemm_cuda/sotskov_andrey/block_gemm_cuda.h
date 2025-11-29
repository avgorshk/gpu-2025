#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& matrixA,
                                 const std::vector<float>& matrixB,
                                 int matrixSize);

#endif // __BLOCK_GEMM_CUDA_H