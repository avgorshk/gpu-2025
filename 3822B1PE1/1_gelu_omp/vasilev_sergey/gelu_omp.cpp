#include "gelu_omp.h"

#include <vector>
#include <cmath>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    if (n == 0) {
        return {};
    }

    std::vector<float> output(n);

    constexpr float kAlpha     = 0.044715f;
    constexpr float kTwoOverPi = 0.63661977236758134308f;
    constexpr float kTwo       = 2.0f;
    constexpr float kOne       = 1.0f;
    constexpr float kHalf      = 0.5f;

    #pragma omp parallel for simd schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        float x  = input[i];
        float x2 = x * x;
        float x3 = x2 * x;

        float inner = kTwoOverPi * (x + kAlpha * x3);

        float e = std::exp(-kTwo * inner);
        float t = (kOne - e) / (kOne + e);

        output[i] = kHalf * x * (kOne + t);
    }

    return output;
}
