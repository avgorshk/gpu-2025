#include "gelu_omp.h"
#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());

    constexpr float coef = 0.044715f;
    constexpr float scale = 0.7978845608f;

#pragma omp parallel for schedule(static) 
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = scale * (x + coef * x3);
        result[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

    return result;
}