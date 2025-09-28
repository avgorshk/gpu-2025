//
// Created by korablev-vm on 27.09.2025.
//

#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> out(input.size());

    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float GELU_COEFF     = 0.044715f;

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        float x  = input[i];
        float x2 = x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x2);

        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

    return out;
}
