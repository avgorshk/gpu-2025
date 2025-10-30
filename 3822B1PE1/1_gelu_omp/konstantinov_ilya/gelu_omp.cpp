#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
  const size_t n = input.size();
  std::vector<float> output(n);

  const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
  const float coeff = 0.044715f;

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    float x = input[i];
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    float tanh_val = std::tanh(inner);
    output[i] = 0.5f * x * (1.0f + tanh_val);
  }

  return output;
}