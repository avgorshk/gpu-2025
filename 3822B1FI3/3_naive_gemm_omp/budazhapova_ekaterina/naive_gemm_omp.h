#ifndef NAIVE_GEMM_OMP_H
#define NAIVE_GEMM_OMP_H

#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
  const std::vector<float>& b,
  int n);

#endif // NAIVE_GEMM_OMP_H