#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> result(size);
    
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float coeff = 0.044715f;
    constexpr float half = 0.5f;
    constexpr float two = 2.0f;
    
    const float* input_ptr = input.data();
    float* result_ptr = result.data();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float x = input_ptr[i];
        float x2 = x * x;
        float x3 = x2 * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        float exp_2z = std::exp(two * inner);
        float tanh_val = (exp_2z - 1.0f) / (exp_2z + 1.0f);
        
        result_ptr[i] = half * x * (1.0f + tanh_val);
    }
    
    return result;
}

