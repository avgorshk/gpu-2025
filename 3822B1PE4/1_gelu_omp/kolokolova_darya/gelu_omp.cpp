#include "gelu_omp.h"
#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    const float pi = 3.14159265358979323846f;
    const float sqrt_2_over_pi = std::sqrt(2.0f / pi);
    const float coefficient = 0.044715f;
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float inner = x + coefficient * x * x * x;
        float tanh_value = std::tanh(sqrt_2_over_pi * inner);
        result[i] = 0.5f * x * (1.0f + tanh_value);
    }
    return result;
}
