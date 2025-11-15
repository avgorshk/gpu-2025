// Copyright (c) 2025 Ionova-Ekaterina
#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  if (input.empty()) return {};

  /**
   * PI = 3.14159
   * constOne = 2 * sqrt(2 / PI)
   * constTwo = constOne * 0.044715
   * tmp = x * (constOne + x * x * constTwo)
   * result = x - x / (1 + exp(tmp))
   */

  constexpr float constOne = 1.595769122f;
  constexpr float constTwo = constOne * 0.044715f;

  auto size = input.size();
  std::vector<float> output(size);

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    float val = input[i];
    float tmp = val * (constOne + val * val * constTwo);
    output[i] = val - val / (1.0f + std::exp(tmp));
  }

  return output;
}