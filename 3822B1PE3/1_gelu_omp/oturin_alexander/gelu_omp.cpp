#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
  std::vector<float> output(input.size());

  float sqrt_2_pi = std::sqrt(2 / std::numbers::pi);

#pragma omp parallel for
  for (size_t i = 0; i < input.size(); i++) {
    float x = input[i];
    float x_cubed = x * x * x;
    output[i] =
        0.5f * x * (1.0f + std::tanh(sqrt_2_pi * (x + 0.044715f * x_cubed)));
  }

  return output;
}
