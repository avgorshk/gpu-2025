#ifndef __BLOCK_GEMM_OMP_H
#define __BLOCK_GEMM_OMP_H

#include <vector>

void matrixMultiplication(const std::vector<float>& a,
						  const std::vector<float>& b,
						  std::vector<float>& result,
						  int iIndex,
						  int jIndex,
						  int step);

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n);

#endif // __BLOCK_GEMM_OMP_H