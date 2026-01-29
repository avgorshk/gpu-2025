#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>
#include <algorithm>

std::vector<float> GeluOMP(const std::vector<float> &input)
{
    std::vector<float> result(input.size());

    static constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    static constexpr float GELU_COEFF = 0.044715f;
    static constexpr float HALF = 0.5f;
    static constexpr float ONE = 1.0f;
    static constexpr float TWO = 2.0f;

    const size_t n = input.size();
    const float *input_ptr = input.data();
    float *result_ptr = result.data();

#pragma omp parallel for schedule(static, 16)
    for (size_t i = 0; i < n; ++i)
    {
        float x = input_ptr[i];
        float x_sq = x * x;
        float x_cubed = x_sq * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
        float exp_2x = std::exp(TWO * inner);
        float tanh_val = ONE - TWO / (exp_2x + ONE);
        result_ptr[i] = HALF * x * (ONE + tanh_val);
    }

    return result;
}