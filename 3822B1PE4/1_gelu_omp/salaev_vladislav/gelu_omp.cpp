#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input)
{
    const size_t n = input.size();
    std::vector<float> output(n);

    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/π)
    const float coeff = 0.044715f;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(n); ++i)
    {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);

        float exp2x = std::exp(2.0f * inner);
        float tanh_val = (exp2x - 1.0f) / (exp2x + 1.0f);

        output[i] = 0.5f * x * (1.0f + tanh_val);
    }

    return output;
}