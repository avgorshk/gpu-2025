#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include "omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const float pi = 3.14159265358979323846f;
    const float sqrt_2_pi = std::sqrt(2.0f / pi);
    std::vector<float> output(input.size());

#pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_pi * (x + 0.044715f * x_cubed);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }

    return output;
}
