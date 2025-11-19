#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();

    std::vector<float> output(n);

    const float* in = input.data();
    float* out = output.data();

    auto gelu = [](float x) -> float {
        const float c = 0.044715f;
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float x3 = x * x * x;
        float z = sqrt_2_over_pi * (x + c * x3);
        float s = 1.0f / (1.0f + std::exp(-2.0f * z));
        return x * s;
    };

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        out[i] = gelu(in[i]);
    }

    return output;
}