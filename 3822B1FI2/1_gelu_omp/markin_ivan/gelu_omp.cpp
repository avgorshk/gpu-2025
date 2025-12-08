#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> result(n);

    if (n == 0) return result;
    
    const float* in = input.data();
    float* out = result.data();
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float t = 0.7978845608028654f * x + 0.035677f * x * x * x;
        out[i] = 0.5f * x * (1.0f + t * (27.0f + t*t) / (27.0f + 9.0f * t*t));
    }
    return result;
}