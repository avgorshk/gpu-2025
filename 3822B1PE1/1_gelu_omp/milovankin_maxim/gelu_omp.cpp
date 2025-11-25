#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);

    const float c = 0.044715f;
    const float sqrt_2_over_pi = 0.7978845608f;

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + c * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
    }

    return output;
}
