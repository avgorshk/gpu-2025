#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline float fast_tanh_like(float z) {
    const float e = std::exp(2.0f * z);
    return (e - 1.0f) / (e + 1.0f);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> out(n);

    constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kCubic = 0.044715f;

#pragma omp parallel for simd schedule(static) if(n > 1024)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        const float x = input[static_cast<std::size_t>(i)];
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float t = kSqrt2OverPi * (x + kCubic * x3);
        const float th = fast_tanh_like(t);
        out[static_cast<std::size_t>(i)] = 0.5f * x * (1.0f + th);
    }
    return out;
}
