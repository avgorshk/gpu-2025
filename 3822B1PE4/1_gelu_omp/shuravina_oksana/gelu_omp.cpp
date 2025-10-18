#include "gelu_omp.h"
#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coefficient = 0.044715f;
    
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        float x = input[i];
        float x_cubed = x * x * x;
        float inner = x + coefficient * x_cubed;
        float tanh_value = std::tanh(sqrt_2_over_pi * inner);
        result[i] = 0.5f * x * (1.0f + tanh_value);
    }
    return result;
}