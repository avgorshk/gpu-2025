#include "gelu_omp.h"

constexpr float sqrt_coeff = 0.7978845608f;
constexpr float half_coeff = 0.5f;
constexpr float two_coeff = 2.0f;
constexpr float one_coeff = 1.0f;
constexpr float special_coeff = 0.044715f;
constexpr float negative_two_coeff = -2.0f;

std::vector<float> GeluOMP(const std::vector<float> &input) {
  size_t size_data = input.size();
  std::vector<float> output(size_data);

#pragma omp parallel for
  for (size_t i = 0; i < size_data; i++) {
    float x = input[i];
    float x_cube = x * x * x;
    float arg_tanh = sqrt_coeff * (x + special_coeff * x_cube);
    float exp_func = two_coeff / (one_coeff + expf(negative_two_coeff * arg_tanh));
    output[i] = half_coeff * x * exp_func;
  }
  return output;
}