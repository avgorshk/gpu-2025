#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

#define M_SQRT2 1.41421356237309504880
#define M_2_SQRTPI 1.12837916709551257390  
#define GELU_COEF 0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535588f 

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> result(n);
    
    const float sqrt_2_over_pi = SQRT_2_OVER_PI;
    const float coef = GELU_COEF;
    const float half = 0.5f;
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; i += 4) {
            size_t limit = (i + 4 < n) ? i + 4 : n;
            
            for (size_t j = i; j < limit; ++j) {
                float x = input[j];
                float x3 = x * x * x;
                float inner = sqrt_2_over_pi * (x + coef * x3);
                float exp_val = expf(2.0f * inner);
                float tanh_val = 1.0f - 2.0f / (exp_val + 1.0f);
                
                result[j] = half * x * (1.0f + tanh_val);
            }
        }
    }
    
    return result;
}
