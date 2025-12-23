#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

static constexpr float GELU_SQRT_2_OVER_PI = 0.7978845608028654f;
static constexpr float GELU_COEFF = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);

    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const float x  = input[i];
        const float x2 = x * x;
        const float x3 = x2 * x;

        const float t = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);

        const float tanh_t = std::tanh(t);

        output[i] = 0.5f * x * (1.0f + tanh_t);
    }

    return output;
}
