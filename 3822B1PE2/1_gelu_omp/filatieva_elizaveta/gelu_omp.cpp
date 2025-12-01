#include "gelu_omp.h"
#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const float pi_val = 3.141592653589793f;
    const float scale_factor = 0.7978845608028654f;
    const float coeff = 0.044715f;
    std::vector<float> result(input.size());
    size_t n = input.size();
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; ++idx) {
        float value = input[idx];
        float cube = value * value * value;
        float transformed = scale_factor * (value + coeff * cube);
        float exp_val = std::exp(2.0f * transformed);
        float tanh_approx = (exp_val - 1.0f) / (exp_val + 1.0f);
        result[idx] = 0.5f * value * (1.0f + tanh_approx);
    }
    return result;
}