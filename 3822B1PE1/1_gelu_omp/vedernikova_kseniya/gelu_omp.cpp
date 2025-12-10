#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t input_size = input.size();
    std::vector<float> output(input_size);

    const float c = 0.044715f;
    const float a = 0.79788456f;  // (2 / pi) ^ (1 / 2)

    #pragma omp parallel for
    for (size_t i = 0; i < input_size; ++i) {
        float x = input[i];
        float tanh_arg = a * x * (1.0f + c * x * x);
        float tanh = (exp(tanh_arg) - (1 / exp(tanh_arg))) / (exp(tanh_arg) + (1 / exp(tanh_arg)));
        output[i] = 0.5f * x * (1.0f + tanh);
    }

    return output;
}