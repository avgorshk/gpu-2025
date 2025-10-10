#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> out(n);

    if (n == 0) return out;

    constexpr float k_sqrt2_over_pi = 0.7978845608f;
    constexpr float k_cubic_coeff   = 0.044715f;
    constexpr float half            = 0.5f;

    const float* in_ptr  = input.data();
    float* out_ptr       = out.data();

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
        float x  = in_ptr[i];
        float x3 = x * x * x;
        float y  = k_sqrt2_over_pi * (x + k_cubic_coeff * x3);
        float t  = tanhf(y);
        out_ptr[i] = half * x * (1.0f + t);
    }

    return out;
}
