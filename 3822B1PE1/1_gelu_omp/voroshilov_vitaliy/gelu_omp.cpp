#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    const float sqrt2pi = sqrtf(2.0 / M_PI);

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        float x3 = x * x * x;
        output[i] = 0.5 * x * (1 + tanhf(sqrt2pi * (x + 0.044715 * x3)));
    }

    return output;
}
