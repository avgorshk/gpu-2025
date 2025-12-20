#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> result(size);
    
    const float sqrt_2_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    const float half = 0.5f;
    const float one = 1.0f;
    const float two = 2.0f;
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < size; ++i) {
            float x = input[i];
            float x3 = x * x * x;
            float inner = sqrt_2_pi * (x + coeff * x3);
            
            float exp_2z = std::exp(two * inner);
            float tanh_val = (exp_2z - one) / (exp_2z + one);
            
            result[i] = half * x * (one + tanh_val);
        }
    }
    
    return result;
}

