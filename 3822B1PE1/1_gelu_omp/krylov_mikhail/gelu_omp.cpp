#include "gelu_omp.h"

#include <immintrin.h>
#include <omp.h>

#include <cmath>
#include <vector>

namespace {
float ApproxExp(float x) {
  return (362880 +
          x * (362880 + x * (181440 + x * (60480 + x * (15120 + x * (3024 + x * (504 + x * (72 + x * (9 + x))))))))) *
         2.75573192e-6;
}

template <typename Vec>
void ResizeUninitialized(Vec& v, std::size_t size) {
  struct stub {
    typename Vec::value_type v;
    stub() {}
  };
  reinterpret_cast<std::vector<stub>&>(v).resize(size);
}
}  // namespace

std::vector<float> GeluOMP(const std::vector<float>& input) {
  static const float kSQRT2DPI = std::sqrt(2.f / M_PI);

  const std::size_t n = input.size();
  std::vector<float> output;
  ResizeUninitialized(output, input.size());

  const float* __restrict__ p_input = input.data();
  float* __restrict__ p_output = output.data();

#pragma omp parallel for simd schedule(static)
  for (std::size_t i = 0; i < n; ++i) {
    const auto x{p_input[i]};
    p_output[i] = x / (1.f + ApproxExp(-2.f * kSQRT2DPI * (x + 0.044715f * (x * x * x))));
  }

  return output;
}