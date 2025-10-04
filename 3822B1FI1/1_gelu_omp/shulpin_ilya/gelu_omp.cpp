#include "gelu_omp.h"

#if defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT __restrict__
#endif

#if defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT __restrict__
#endif

static inline float gelu_scalar_exp(float x) {
    constexpr float c = 0.044715f;
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    float x3 = x * x * x;
    float z = sqrt_2_over_pi * (x + c * x3);
    // GELU ≈ x * sigmoid(2*z) = x / (1 + exp(-2*z))
    float s = 1.0f / (1.0f + std::exp(-2.0f * z));
    return x * s;
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    std::vector<float> output(n);

    const float* RESTRICT in = input.data();
    float* RESTRICT out = output.data();

    const size_t limit = (n / 4) * 4;
    const size_t chunks = limit / 4;

    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < chunks; ++t) {
        size_t base = t * 4;
        out[base + 0] = gelu_scalar_exp(in[base + 0]);
        out[base + 1] = gelu_scalar_exp(in[base + 1]);
        out[base + 2] = gelu_scalar_exp(in[base + 2]);
        out[base + 3] = gelu_scalar_exp(in[base + 3]);
    }

    for (size_t i = limit; i < n; ++i) {
        out[i] = gelu_scalar_exp(in[i]);
    }

    return output;
}