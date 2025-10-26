#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline float fast_tanhf(float z) {
  if (z > 10.0f) return 1.0f;
  if (z < -10.0f) return -1.0f;
  float e = std::exp(-2.0f * z);
  return (1.0f - e) / (1.0f + e);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
  const std::size_t n = input.size();
  std::vector<float> out(n);
  if (n == 0) return out;


  constexpr float K = 0.7978845608f;
  constexpr float CUBIC = 0.044715f;
  constexpr float HALF = 0.5f;
  constexpr float ONE = 1.0f;


#pragma omp parallel for schedule(static)
  for (long long i = 0; i < static_cast<long long>(n); ++i) {
    float x = input[static_cast<std::size_t>(i)];
    float x2 = x * x;
    float x3 = x2 * x;
    float s = K * (x + CUBIC * x3);
    float t = fast_tanhf(s);
    out[static_cast<std::size_t>(i)] = HALF * x * (ONE + t);
  }

  return out;
}
