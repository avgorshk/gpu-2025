// Copyright (c) 2025 Ionova-Ekaterina
#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int size) {
  auto countElem = size * size;
  if (a.size() != countElem || b.size() != countElem) return {};

  std::vector<float> c(countElem, 0.0f);
  constexpr auto blockSize = 8;
  auto numBlocks = size / blockSize;

#pragma omp parallel for shared(a, b, c)
  for (int i = 0; i < numBlocks; ++i)
    for (int j = 0; j < numBlocks; ++j)
      for (int block = 0; block < numBlocks; ++block)
        for (int m = i * blockSize; m < (i + 1) * blockSize; ++m)
          for (int n = j * blockSize; n < (j + 1) * blockSize; ++n)
            for (int k = block * blockSize; k < (block + 1) * blockSize; ++k)
              c[m * size + n] += a[m * size + k] * b[k * size + n];

  return c;
}