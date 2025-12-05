#include <omp.h>

#include <cmath>
#include <vector>

#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
  const size_t n = input.size();
  std::vector<float> output(n);

  constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
  constexpr float COEFF = 0.044715f;
  constexpr float HALF = 0.5f;

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    float x = input[i];
    float x2 = x * x;
    float x3 = x * x2;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    float t = std::tanh(inner);
    output[i] = HALF * x * (1.0f + t);
  }

  return output;
}
