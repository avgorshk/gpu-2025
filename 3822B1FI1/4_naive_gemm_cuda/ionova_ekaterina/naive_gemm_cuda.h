// Copyright (c) 2025 Ionova-Ekaterina
#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int size);

#endif  // __NAIVE_GEMM_CUDA_H