#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include "omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const float pi = 3.14159265358979323846f;
    const float sqrt_2_pi = std::sqrt(2.0f / pi);

#pragma omp parallel for
    for (size_t pos = 0; pos < input.size(); ++pos) {
        const float in_val = input[pos];
        const float cubic = in_val * in_val * in_val;
        const float transformed = sqrt_2_pi * (in_val + 0.044715f * cubic);
        output[pos] = 0.5f * in_val * (1.0f + std::tanh(transformed));
    }

    return output;
}
