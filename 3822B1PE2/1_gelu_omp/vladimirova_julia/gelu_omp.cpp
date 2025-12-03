#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t vector_size = input.size();
    std::vector<float> output(vector_size);
    float ct_1 = sqrt(2.0f/acos(-1.0f));
    float ct_2 = 0.044715f;
#pragma omp parallel for
    for (int i = 0; i < vector_size; i++) {
        float x = input[i];
        output[i] = x * 0.5f * (1.0f + tanh(ct_1 * x * (1.0f + ct_2 * x * x)));
    }
    return output;
}
