#include "gelu_omp.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

std::vector<float> GeluOMP(const std::vector<float> &input) {
  const int n = static_cast<int>(input.size()); // <-- int вместо size_t
  std::vector<float> output(n);

  const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
  const float coeff = 0.044715f;

#pragma omp parallel for
  for (int i = 0; i < n; ++i) { // <-- int i
    float x = input[i];
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    output[i] = 0.5f * x * (1.0f + std::tanh(inner));
  }

  return output;
}