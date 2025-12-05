#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input) {
    int count = input.size();
    std::vector<float> res(count);
#pragma omp parallel for simd
    for (int i = 0; i < count; ++i) {
        const float x = input[i];
        const float arg = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        const float tanh = std::tanh(arg);
        res[i] = 0.5f * x * (1.0f + tanh);
    }
    return res;
}