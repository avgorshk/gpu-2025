//
// Created by egorm on 04-Oct-25.
//
#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input) {
    size_t size = input.size();
    std::vector<float> res(size);

    constexpr float a = 0.7978845608028654f;
    constexpr float b = 0.044715f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;
#pragma omp parallel for simd default(none) shared(input, res, size, a, b, half, one)
    for (int i = 0; i < size; ++i) {
        const float x = input[i];
        const float x_cube = x * x * x;
        const float tanh_args = a * (x + b * x_cube);
        const float tanh_result = std::tanh(tanh_args);
        res[i] = half * x * (one + tanh_result);
    }
    return res;
}