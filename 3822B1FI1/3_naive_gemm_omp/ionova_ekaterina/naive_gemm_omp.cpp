// Copyright (c) 2025 Ionova-Ekaterina
#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int size) {
  auto countElem = size * size;
  if (a.size() != countElem || b.size() != countElem) return {};

  std::vector<float> c(countElem, 0.0f);

#pragma omp parallel for shared(a, b, c)
  for (int m = 0; m < size; ++m)
    for (int n = 0; n < size; ++n)
      for (int k = 0; k < size; ++k)
        c[m * size + n] += a[m * size + k] * b[size * k + n];

  return c;
}