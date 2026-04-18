#include "gelu_omp.h"


std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t vector_size = input.size();
    std::vector<float> output(vector_size);

#pragma omp parallel for
    for (int i = 0; i < vector_size; i++) {
        float x = input[i];
        output[i] = x * 0.5f * (1.0f + tanh(sqrt(2.0f/acos(-1.0f)) * x * (1.0f + 0.044715f * x * x)));
    }
    return output;
}
