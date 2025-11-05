#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> result(size);

    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float coeff = 0.044715f;
    constexpr float half = 0.5f;

#pragma omp parallel
    {
#pragma omp for simd schedule(static)
        for (size_t i = 0; i < size; ++i) {
            float x = input[i];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            float exp_val = expf(-2.0f * inner);
            float tanh_val = (1.0f - exp_val) / (1.0f + exp_val);

            result[i] = half * x * (1.0f + tanh_val);
        }
    }

    return result;
}