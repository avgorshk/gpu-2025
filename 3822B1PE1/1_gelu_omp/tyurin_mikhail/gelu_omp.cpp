#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
#include <vector>

static inline float gelu_single(float x) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    float inner = k0 * (x + k1 * x3);
    float e = expf(2.0f * inner);
    float tanh_val = (e - 1.0f) / (e + 1.0f);
    return 0.5f * x * (1.0f + tanh_val);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; ++i) {
        output[i] = gelu_single(input[i]);
    }
    return output;
}

