#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

constexpr float GELU_SCALE_FACTOR = 0.5f;
constexpr float GELU_TANH_SCALE = 0.7978845608028654f;
constexpr float GELU_CUBIC_COEFFICIENT = 0.044715f;

inline float ComputeGeluWithExp(float x) {
    float inner_value = GELU_TANH_SCALE * x * (1.0f + GELU_CUBIC_COEFFICIENT * x * x);
    float exp_value = std::exp(2.0f * inner_value);
    float tanh_value = (exp_value - 1.0f) / (exp_value + 1.0f);
    return GELU_SCALE_FACTOR * x * (1.0f + tanh_value);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        output[i] = ComputeGeluWithExp(input[i]);
    }

    return output;
}
