#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float sqrt_2_over_pi = std::sqrt(2.0f / 3.14159265358979323846f);

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x3);
        float gelu = 0.5f * x * (1.0f + std::tanh(tanh_arg));
        output[i] = gelu;
    }

    return output;
}