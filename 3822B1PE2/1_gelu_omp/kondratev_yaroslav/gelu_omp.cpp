#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> result(size);
    
    const float sqrt_2_pi = 0.7978845608f;  // sqrt(2/π)
    const float coeff = 0.044715f;
    const float half = 0.5f;
    const float one = 1.0f;

    #pragma unroll 4
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_pi * (x + coeff * x3);
        result[i] = half * x * (one + std::tanh(inner));
    }
    
    return result;
}