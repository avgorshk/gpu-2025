// Copyright (c) 2025 Mironov-Ilya
#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int size);

#endif  // __BLOCK_GEMM_CUDA_H