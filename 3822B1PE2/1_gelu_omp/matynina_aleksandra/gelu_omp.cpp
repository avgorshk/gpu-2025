#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    
    if (size == 0) {
        return std::vector<float>();
    }
    
    std::vector<float> output(size);
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;
    constexpr float HALF = 0.5f;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(size); ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float arg = SQRT_2_OVER_PI * (x + COEFF * x3);
        float exp2arg = std::exp(2.0f * arg);
        float tanh_val = (exp2arg - 1.0f) / (exp2arg + 1.0f);
        output[i] = HALF * x * (1.0f + tanh_val);
    }
    
    return output;
}