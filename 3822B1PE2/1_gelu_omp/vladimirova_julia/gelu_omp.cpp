#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t vector_size = input.size();
    std::vector<float> output(input);
#pragma omp parallel for
    for (int i = 0; i < vector_size; i++) {
        output[i]/=(1.0f + exp(-1.701744 * input[i]));
    }
    return output;
}