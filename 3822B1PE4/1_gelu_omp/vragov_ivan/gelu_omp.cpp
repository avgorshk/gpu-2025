#include "gelu_omp.h"
#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);
    
    // Константы для вычисления GELU
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    const float half = 0.5f;
    
    // Параллелизация с векторизацией
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        // Используем exp вместо tanh для лучшей производительности
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        float exp_2x = expf(2.0f * inner);
        float tanh_val = (exp_2x - 1.0f) / (exp_2x + 1.0f);
        output[i] = half * x * (1.0f + tanh_val);
    }
    
    return output;
}

