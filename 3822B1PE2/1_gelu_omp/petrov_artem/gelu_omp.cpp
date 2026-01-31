#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> result(n);
    
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float GELU_COEFF = 0.044715f;
    constexpr float HALF = 0.5f;
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const float x = input[i];
            const float x3 = x * x * x;
            
            const float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
            
            const float exp_2inner = expf(2.0f * inner);
            const float tanh_val = (exp_2inner - 1.0f) / (exp_2inner + 1.0f);
            
            result[i] = HALF * x * (1.0f + tanh_val);
        }
    }
    
    return result;
}