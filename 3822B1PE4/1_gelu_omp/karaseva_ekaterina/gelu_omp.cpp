#include "gelu_omp.h"
#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    const float square = std::sqrt(2.0f / M_PI);
    const float koff = 0.044715f;
    
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        float x = input[i];
        float x_cubed = x * x * x;
        float inner = square * (x + koff * x_cubed);
        float tanh_value = std::tanh(inner);
        result[i] = 0.5f * x * (1.0f + tanh_value);
    }
    return result;
}