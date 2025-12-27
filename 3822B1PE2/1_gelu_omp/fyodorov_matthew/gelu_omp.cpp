#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  std::vector<float> result(input.size());

  const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
  const float coeff = 0.044715f;

#pragma omp parallel for simd
  for (size_t i = 0; i < input.size(); ++i) {
    float x = input[i];
    float x_cubed = x * x * x;

    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);

    float tanh_val = 0.0f;

    if (inner > 4.0f) {
      tanh_val = 1.0f;
    } else if (inner < -4.0f) {
      tanh_val = -1.0f;
    } else {
      float exp2x = expf(2.0f * inner);
      tanh_val = (exp2x - 1.0f) / (exp2x + 1.0f);
    }

    result[i] = 0.5f * x * (1.0f + tanh_val);
  }

  return result;
}