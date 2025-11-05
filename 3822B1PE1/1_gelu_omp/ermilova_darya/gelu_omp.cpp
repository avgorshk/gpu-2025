#include "gelu_omp.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    std::vector<float> output(n);
    const float sqr = std::sqrt(2.0f / M_PI);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        output[i] = 0.5f * x * (1.0f + std::tanh(sqr * (x + 0.044715f * x3)));
    }
    return output;
}
