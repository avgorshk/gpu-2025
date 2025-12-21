#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input) {
    constexpr float SQRT_2_PI = 0.7978845608f; // sqrt(2 / pi)
    constexpr float COEF = 0.044715f;

    size_t size = input.size();
    std::vector<float> output(size);

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++) {
        float x = input[i];
        float arg = SQRT_2_PI * (x + COEF * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(arg));
    }

    return output;
}
