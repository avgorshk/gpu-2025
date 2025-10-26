#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
  std::vector<float> output(input.size());

  constexpr float alpha = 0.044715f;
  constexpr float beta = 0.7978845608028654f;

#pragma omp parallel for
  for (size_t i = 0; i < input.size(); ++i) {
    float x = input[i];
    float inner = beta * (x + alpha * x * x * x);
    output[i] = 0.5f * x * (1.0f + std::tanh(inner));
  }

  return output;
}