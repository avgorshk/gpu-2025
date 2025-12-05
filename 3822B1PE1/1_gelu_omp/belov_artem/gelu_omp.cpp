#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);
    
    constexpr float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
    constexpr float coeff = 0.044715f;
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const float x = input[i];
        const float x_cubed = x * x * x;
        
        const float arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        const float exp_2arg = std::exp(2.0f * arg);
        const float tanh_arg = (exp_2arg - 1.0f) / (exp_2arg + 1.0f);
        
        output[i] = 0.5f * x * (1.0f + tanh_arg);
    }

    return output;
}
