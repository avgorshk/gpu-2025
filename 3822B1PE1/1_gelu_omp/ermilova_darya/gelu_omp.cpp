#include "gelu_omp.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    std::vector<float> output(n);
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    constexpr float GELU_COEFF = 0.044715f;


#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        output[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + GELU_COEFF * x3)));
    }
    return output;
}
