#include "gelu_omp.h"

#include <omp.h>

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
  std::vector<float> output(input.size());
  const double glob_sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
  const double glob_coef = 0.044715;
#pragma omp parallel for
  for (size_t i = 0; i < input.size(); ++i) {
    double x = input[i];
    double inner = glob_sqrt_2_over_pi * (x + glob_coef * (x * x * x));
    output[i] = (0.5) * x * (1.0 + std::tanh(inner));
  }

  return output;
}
