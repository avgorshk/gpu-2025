#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int size = static_cast<int>(input.size());
    std::vector<float> output(size);
    
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
    
    return output;
}

