#include "gelu_omp.h"
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    std::vector<float> output(input.size());
    
#pragma omp parallel
    {
#pragma omp for
        for (int index = 0; index < static_cast<int>(input.size()); ++index)
        {
            float x = input[index];
            float x_cubed = x * x * x;
            float inner_value = constants::sqrt_2_over_pi * 
                               (x + constants::gelu_coefficient * x_cubed);
            output[index] = 0.5f * x * (1.0f + std::tanh(inner_value));
        }
    }
    return output;
}