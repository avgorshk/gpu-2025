#include <cmath>
#include <omp.h>
#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) return {};
    
    std::vector<float> output(input.size());
    const float sqrt2OverPi = std::sqrt(2.0f / acosf(-1.0f));
    constexpr float coefficient = 0.044715f;

#pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float arg = sqrt2OverPi * (x + coefficient * x3);
        float exp2arg = expf(2.0f * arg);
        float tanhValue = (exp2arg - 1.0f) / (exp2arg + 1.0f);
        output[i] = 0.5f * x * (1.0f + tanhValue);
    }

    return output;
}