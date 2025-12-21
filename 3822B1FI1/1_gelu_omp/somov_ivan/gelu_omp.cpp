#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();

    std::vector<float> output(n);

    const float* in = input.data();
    float* out = output.data();

    const size_t limit = (n / 4) * 4;
    const size_t chunks = limit / 4;

    constexpr float c = 0.044715f;
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;

    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < chunks; ++t) {
        size_t base = t * 4;

        {
            float x = in[base + 0];
            float x3 = x * x * x;
            float z = sqrt_2_over_pi * (x + c * x3);
            float s = 1.0f / (1.0f + std::exp(-2.0f * z));
            out[base + 0] = x * s;
        }

        {
            float x = in[base + 1];
            float x3 = x * x * x;
            float z = sqrt_2_over_pi * (x + c * x3);
            float s = 1.0f / (1.0f + std::exp(-2.0f * z));
            out[base + 1] = x * s;
        }

        {
            float x = in[base + 2];
            float x3 = x * x * x;
            float z = sqrt_2_over_pi * (x + c * x3);
            float s = 1.0f / (1.0f + std::exp(-2.0f * z));
            out[base + 2] = x * s;
        }

        {
            float x = in[base + 3];
            float x3 = x * x * x;
            float z = sqrt_2_over_pi * (x + c * x3);
            float s = 1.0f / (1.0f + std::exp(-2.0f * z));
            out[base + 3] = x * s;
        }
    }

    for (size_t i = limit; i < n; ++i) {
        float x = in[i];
        float x3 = x * x * x;
        float z = sqrt_2_over_pi * (x + c * x3);
        float s = 1.0f / (1.0f + std::exp(-2.0f * z));
        out[i] = x * s;
    }

    return output;
}