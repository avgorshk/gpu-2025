
#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float tanh_val = std::tanh(std::sqrt(2.0f / M_PI) * input[i] * (1 + 0.044715f * std::pow(input[i],2)));
        output[i] = 0.5f * input[i] * (1.0f + tanh_val);
    }
    return output;
}
