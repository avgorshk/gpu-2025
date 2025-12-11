#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
constexpr float COEF = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> output(size);

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEF * x3);
        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

    return output;
}
