#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
  std::vector<float> out(input.size());
  constexpr float SQRT_2_OVER_PI = -2.0f * std::sqrt(2.0f / M_PI);

#pragma omp parallel for
  for (size_t i = 0; i < input.size(); i++) {
    float x = input[i];
    // https://en.wikipedia.org/wiki/Logistic_function#Hyperbolic_tangent
    out[i] = x / (1.0f + std::exp(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
  }

  return out;
}
