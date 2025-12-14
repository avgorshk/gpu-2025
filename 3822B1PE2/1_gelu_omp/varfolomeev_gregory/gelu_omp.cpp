#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) return {};
    
    std::vector<float> output(n);
    
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;
    
    const float* __restrict src = input.data();
    float* __restrict dst = output.data();

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        float x = src[i];
        float x3 = x * x * x;
        float t = kSqrt2OverPi * (x + kCoeff * x3);
        float exp2t = std::exp(2.0f * t);
        float tanh_t = (exp2t - 1.0f) / (exp2t + 1.0f);
        dst[i] = 0.5f * x * (1.0f + tanh_t);
    }
    
    return output;
}

