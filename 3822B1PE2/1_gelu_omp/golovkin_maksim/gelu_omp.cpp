#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    int size = static_cast<int>(input.size());
    std::vector<float> output(size);

    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoeff = 0.044715f;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }

    return output;
}
