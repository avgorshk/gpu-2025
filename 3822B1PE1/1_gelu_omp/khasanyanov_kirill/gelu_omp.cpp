#include "gelu_omp.h"

#include <cmath>
#include <vector>

static inline float fast_tanhf(float x) {
  if (x > 5.0f)
    return 1.0f;
  if (x < -5.0f)
    return -1.0f;
  const float e = std::exp(-2.0f * x);
  return (1.0f - e) / (1.0f + e);
}

std::vector<float> GeluOMP(const std::vector<float> &input) {
  const size_t n = input.size();
  std::vector<float> output(n);
  const float k = 0.7978845608f; // sqrt(2/pi)
  const float c = 0.044715f;

#pragma omp parallel for schedule(static)
  for (long long i = 0; i < static_cast<long long>(n); ++i) {
    const float x = input[i], x2 = x * x, x3 = x2 * x;
    const float t = k * (x + c * x3);
    const float th = fast_tanhf(t);
    output[i] = 0.5f * x * (1.0f + th);
  }

  return output;
}
