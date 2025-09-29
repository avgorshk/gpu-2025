#include "kudryashova_gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int total_elements = static_cast<int>(input.size());

    std::vector<float> output(total_elements);

    constexpr float coeff = 0.044715f;
    const float sqrt_pi = std::sqrt(2.0f / std::acos(-1.0f));

#pragma omp parallel for
    for (int index = 0; index < total_elements; ++index) {
        float input_value = input[index];
        float cub = input_value * input_value * input_value;

        float arg_tanh = sqrt_pi * (input_value + coeff * cub);

        float exp_term = std::expf(2.0f * arg_tanh);
        float tanh = (exp_term - 1.0f) / (exp_term + 1.0f);

        float gelu_result = 0.5f * input_value * (1.0f + tanh);

        output[index] = gelu_result;
    }
    return output;
}