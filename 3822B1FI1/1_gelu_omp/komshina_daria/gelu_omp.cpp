#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> out;
    out.resize(n);

    const float k0 = 0.5f;
    const float k1 = 0.7978845608028654f;
    const float k2 = 0.044715f;

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        float x = input[static_cast<std::size_t>(i)];
        float x3 = x * x * x;
        float z = k1 * (x + k2 * x3);

        float e = std::exp(2.0f * z);
        float tanh_z = (e - 1.0f) / (e + 1.0f);

        out[static_cast<std::size_t>(i)] = k0 * x * (1.0f + tanh_z);
    }

    return out;
}
