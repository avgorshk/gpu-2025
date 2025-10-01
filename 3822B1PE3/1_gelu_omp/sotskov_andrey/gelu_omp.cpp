#include "gelu_omp.h"

#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input) {
    const size_t size = input.size();
    if (size == 0)
    return {};

    std::vector<float> result(size);

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
    const float x = input[i];
    const float x_cubed = x * x * x;

    const float inner = sqrt_2_over_pi * (x + coef * x_cubed);
    const float tanh_val = 1.0f - 2.0f / (std::exp(2.0f * inner) + 1.0f);

    result[i] = 0.5f * x * (1.0f + tanh_val);
    }

    return result;
}