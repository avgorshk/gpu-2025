#include <omp.h>
#include <cmath>
#include "gelu_omp.h"

AlignedVector GeluOMP(const AlignedVector &input)
{
    AlignedVector result(input.size());
    const float precalc_sqrt = std::sqrt(2.0f / M_PI);
    float x = 0;
    float x3 = 0;
    float arg = 0;
    auto sz = input.size();
#pragma omp parallel for
    for (size_t i = 0; i < sz; i++)
    {
        x = input[i];
        x3 = x * x * x;
        arg = precalc_sqrt * (x + 0.044715f * x3);
        result[i] = 0.5f * x * (1.0f + std::tanh(arg));
    }
    return result;
}