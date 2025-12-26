#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input)
{
    std::vector<float> result(input.size());

    float sqrt_2_over_pi = sqrt(2.0f / acos(-1.0f));
    const float coeff = 0.044715f;

#pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i)
    {
        float x = input[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float exparg = expf(2.0f * inner);
        float sigmoid = (exparg - 1.0f) / (exparg + 1.0f);
        result[i] = 0.5f * x * (1.0f + sigmoid);
    }

    return result;
}