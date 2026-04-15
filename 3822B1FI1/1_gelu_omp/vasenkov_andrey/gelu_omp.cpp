#include "gelu_omp.h"


std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t vector_size = input.size();
    std::vector<float> output(input);
    const float* in_ptr = input.data();
    float* out_ptr = output.data();

#pragma omp parallel for schedule(static, 8388608)
    for (int i = 0; i < static_cast<int>(vector_size); i++) {
        out_ptr[i] /= (1.0f + std::exp(-1.6f * in_ptr[i]));
    }

    return output;
}
