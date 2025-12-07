#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input)
{
    const std::size_t n = input.size();
    std::vector<float> output(n);

    if (n == 0)
    {
        return output;
    }

    constexpr float kSqrt2OverPiScaled = 1.595769122f;
    constexpr float kC = 0.044715f;

    const float *in = input.data();
    float *out = output.data();

#pragma omp parallel for simd schedule(static)
    for (std::size_t i = 0; i < n; ++i)
    {
        const float x = in[i];
        const float x_cubed = x * x * x;
        const float z = kSqrt2OverPiScaled * (x + kC * x_cubed);
        const float s = 1.0f / (1.0f + std::exp(-z));
        out[i] = x * s;
    }

    return output;
}
