#ifndef __BLOCK_GEMM_OMP_H
#define __BLOCK_GEMM_OMP_H

#include <omp.h>
#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n);

#endif // __BLOCK_GEMM_OMP_H