#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
  std::vector<float> output(input.size());
  const float sqrt_2_pi_m = std::sqrt(2.0f / M_PI) * 2;

#pragma omp parallel for simd
  for (size_t i = 0; i < input.size(); i++) {
    const float x = input[i];
    const float x_cubed = x * x * x;
    const float expo = std::exp(sqrt_2_pi_m * (x + 0.044715f * x_cubed));
    output[i] = x * expo / (expo + 1.0f);
  }

  return output;
}
