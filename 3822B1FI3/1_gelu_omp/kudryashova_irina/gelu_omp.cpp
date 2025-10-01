#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input)
{
    std::vector<float> output(input.size());
    const float sqrt_pi = std::sqrt(2.0f / std::acos(-1.0f));
#pragma omp parallel for
    for (int index = 0; index < input.size(); ++index)
    {
        float input_value = input[index];
        float arg_tanh = sqrt_pi * (input_value + 0.044715f * input_value * input_value * input_value);
        float exp_term = std::exp(2.0f * arg_tanh);
        float tanh = (exp_term - 1.0f) / (exp_term + 1.0f);
        output[index] = 0.5f * input_value * (1.0f + tanh);
    }
    return output;
}
