#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input)
{
    std::vector<float> result(input.size());

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

#pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i)
    {
        float x = input[i];
        float x_cubed = x * x * x;

        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float sigmoid = 1.0f / (1.0f + std::exp(-2.0f * inner));

        result[i] = 0.5f * x * (1.0f + sigmoid);
    }

    return result;
}
