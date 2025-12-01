#include "gelu_omp.h"
#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    constexpr float PI_CONST = 3.14159265358979323846f;
    constexpr float SQRT_2_OVER_PI = 0.79788458347320556640625f;
    constexpr float GELU_COEFF = 0.044715f;

    std::vector<float> output(input.size());
    const int total_elements = static_cast<int>(input.size());

#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_elements; ++i) {
        float x = input[i];
        float x_cubed = x * x * x;

        float inner_expr = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);

        float exp_positive = std::exp(inner_expr);
        float exp_negative = std::exp(-inner_expr);
        float tanh_approx = (exp_positive - exp_negative) / (exp_positive + exp_negative);

        output[i] = 0.5f * x * (1.0f + tanh_approx);
    }

    return output;
}