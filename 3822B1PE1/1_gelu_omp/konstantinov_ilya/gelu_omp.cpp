#include "gelu_omp.h"
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
  const size_t size = input.size();
  std::vector<float> output(size);

  if (size == 0) {
    return output;
  }

  constexpr float alpha = 0.044715f;
  constexpr float beta = 0.7978845608028654f;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; ++i) {
    const float x = input[i];
    const float x_cube = x * x * x;
    const float inner = beta * (x + alpha * x_cube);
    const float exp_val = std::exp(-2.0f * inner);
    const float tanh_approx = 1.0f - 2.0f / (1.0f + exp_val);
    output[i] = 0.5f * x * (1.0f + tanh_approx);
  }

  return output;
}