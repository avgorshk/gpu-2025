#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size(), 0.0f);

    static constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    static constexpr float GELU_COEFF = 0.044715f;

    const float* input_ptr = input.data();
    float* result_ptr = result.data();
    int input_size = static_cast<int>(input.size());

    #pragma omp parallel for schedule(dynamic, 4096)
    for (int idx = 0; idx < input_size; ++idx) {
        float x_val = input_ptr[idx];
        float x_cubed = x_val * x_val * x_val;
        float polynomial_term = x_val + GELU_COEFF * x_cubed;
        float tanh_arg = SQRT_2_OVER_PI * polynomial_term;

        float exp_2x = std::exp(2.0f * tanh_arg);
        float tanh_result = (exp_2x - 1.0f) / (exp_2x + 1.0f);
        result_ptr[idx] = 0.5f * x_val * (1.0f + tanh_result);
    }
    
    return result;
}