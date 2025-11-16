#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output;
    output.resize(n);
    if (n == 0) return output;

    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float coeff = 0.044715f;

    const float* __restrict src = input.data();
    float* __restrict dst = output.data();

#pragma omp parallel
    {
#pragma omp for simd schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            float x = src[static_cast<std::size_t>(i)];
            float x3 = x * x * x;
            float t = sqrt_2_over_pi * (x + coeff * x3);

            float tanh_t;
            if (t >= 0.0f) {
                float z = expf(-2.0f * t);
                tanh_t = (1.0f - z) / (1.0f + z);
            }
            else {
                float z = expf(2.0f * t);
                tanh_t = (z - 1.0f) / (z + 1.0f);
            }

            dst[static_cast<std::size_t>(i)] = 0.5f * x * (1.0f + tanh_t);
        }
    }

    return output;
}
