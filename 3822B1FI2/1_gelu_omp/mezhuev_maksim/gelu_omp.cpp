#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  std::vector<float> output(input.size());

  constexpr float sqrt_2_pi = 0.7978845608028654f;
  constexpr float coeff = 0.044715f;
  constexpr float half = 0.5f;
  const size_t n = input.size();

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    float x = input[i];
    float x3 = x * x * x;
    float inner = sqrt_2_pi * (x + coeff * x3);
    float exp_val = std::exp(2.0f * inner);
    float tanh_val = (exp_val - 1.0f) / (exp_val + 1.0f);

    output[i] = half * x * (1.0f + tanh_val);
  }

  return output;
}