#include <omp.h>

#include <cstddef>
#include <vector>

#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n) {
  const size_t N = static_cast<size_t>(n);
  const size_t NN = N * N;

  std::vector<float> c(NN, 0.0f);

  const float* a_data = a.data();
  const float* b_data = b.data();
  float* c_data = c.data();

#pragma omp parallel for schedule(static)
  for (long long i = 0; i < static_cast<long long>(N); ++i) {
    float* c_row = c_data + i * N;

    for (size_t k = 0; k < N; ++k) {
      const float a_ik = a_data[i * N + k];
      const float* b_row = b_data + k * N;

      size_t j = 0;
      size_t limit = N & ~static_cast<size_t>(3);

      for (; j < limit; j += 4) {
        c_row[j] += a_ik * b_row[j];
        c_row[j + 1] += a_ik * b_row[j + 1];
        c_row[j + 2] += a_ik * b_row[j + 2];
        c_row[j + 3] += a_ik * b_row[j + 3];
      }

      for (; j < N; ++j) {
        c_row[j] += a_ik * b_row[j];
      }
    }
  }

  return c;
}

