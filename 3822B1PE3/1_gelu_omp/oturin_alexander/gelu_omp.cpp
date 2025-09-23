#include "gelu_omp.hpp"

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

inline float fast_tanh(float x) {
  const float x2 = x * x;
  return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

std::vector<float> GeluOMPapprox(const std::vector<float>& input) {
  std::vector<float> output(input.size());
  const float sqrt_2_pi = std::sqrt(2.0f / std::numbers::pi_v<float>);

#pragma omp parallel for simd
  for (size_t i = 0; i < input.size(); i++) {
    const float x = input[i];
    const float x_cubed = x * x * x;
    const float alpha = sqrt_2_pi * (x + 0.044715f * x_cubed);
    output[i] = 0.5f * x * (1.0f + fast_tanh(alpha));
  }

  return output;
}
