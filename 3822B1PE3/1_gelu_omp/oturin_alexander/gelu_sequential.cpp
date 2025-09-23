#include "gelu_sequential.hpp"

std::vector<float> GeluSEQ(const std::vector<float>& input) {
  std::vector<float> output(input.size());

  for (size_t i = 0; i < input.size(); i++) {
    float x = input[i];
    output[i] = 0.5 * x *
                (1 + std::tanh(std::sqrt(2 / std::numbers::pi) *
                               (x + 0.044715 * std::pow(x, 3))));
  }

  return output;
}
