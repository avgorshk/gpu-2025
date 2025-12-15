#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    constexpr float kAlpha = 0.044715f;
    constexpr float kSqrt2toPi = 0.7978845608028654f;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float x2 = x * x;
        float x3 = x2 * x;

        float t = kSqrt2toPi * (x + kAlpha * x3);

        float exp_term = expf(-2.0f * t);
        float tanh_approx = 2.0f / (1.0f + exp_term) - 1.0f;

        output[i] = 0.5f * x * (1.0f + tanh_approx);
    }

    return output;
}