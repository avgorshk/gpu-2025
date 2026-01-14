#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> result(n);
    
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float GELU_COEFF = 0.044715f;
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const float x = input[i];
            const float z = 1.702f * x;
            const float exp_neg_z = expf(-z);
            result[i] = x / (1.0f + exp_neg_z);
        }
    }
    
    return result;
}
