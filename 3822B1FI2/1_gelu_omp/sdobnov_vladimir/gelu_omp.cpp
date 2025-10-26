#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> output(size);

    constexpr float a = 0.7978845608028654f;  // sqrt(2/π)
    constexpr float b = 0.044715f;
    constexpr float half = 0.5f;

#pragma omp parallel
    {
#pragma omp for simd schedule(static) aligned(input, output: 32)
        for (size_t i = 0; i < size; ++i) {
            float x = input[i];
            float x3 = x * x * x;
            float z = a * (x + b * x3);
            float sigmoid = 1.0f / (1.0f + std::exp(-1.702f * z)); // Аппроксимация tanh

            output[i] = half * x * (1.0f + sigmoid);
        }
    }

    return output;
}