#include <cmath>
#include <omp.h>
#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) return {};
    
    std::vector<float> output(input.size());
    const float sqrt_2_pi = std::sqrt(2.0f / acosf(-1.0f));
    constexpr float coeff = 0.044715f;

#pragma omp parallel for
    for (size_t idx = 0; idx < input.size(); ++idx) {
        float x = input[idx];
        float x3 = x * x * x;
        float arg = sqrt_2_pi * (x + coeff * x3);
        float exp2z = expf(2.0f * arg);
        float tanh_val = (exp2z - 1.0f) / (exp2z + 1.0f);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }

    return output;
}