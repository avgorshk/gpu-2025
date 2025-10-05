#include "gelu_omp.h"
#include <cmath>
#include <cstddef>
#include <omp.h>

static inline float tanh_exp(float t) {
    float e2 = std::expf(-2.0f * t);         
    return (1.0f - e2) / (1.0f + e2);         
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> out(n);
    if (n == 0) return out;

    constexpr float kInvSqrtPi2 = 0.7978845608028654f; 
    constexpr float kCubicCoeff = 0.044715f;          

    const float* __restrict src = input.data();
    float* __restrict dst = out.data();

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        float x = src[static_cast<std::size_t>(i)];
        float x3 = x * x * x;
        float t = kInvSqrtPi2 * (x + kCubicCoeff * x3);
        float th = tanh_exp(t);
        dst[static_cast<std::size_t>(i)] = 0.5f * x * (1.0f + th);
    }
    return out;
}
