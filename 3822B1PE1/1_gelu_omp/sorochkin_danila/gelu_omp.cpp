#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());

    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float GELU_COEFF = 0.044715f;
    constexpr float MUL = 2.0f * SQRT_2_OVER_PI;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < input.size(); ++i) {
        const float x = input[i];
        const float x3 = x * x * x;
        const float z = MUL * (x + GELU_COEFF * x3);

        result[i] = x / (1.0f + expf(-z));
    }

    return result;
}