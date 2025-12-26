#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>

constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
constexpr float COEF = 0.044715f;
constexpr float HALF = 0.5f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> output(size);

    const float* input_data = input.data();
    float* output_data = output.data();

    int num_threads = omp_get_max_threads();

#pragma omp parallel num_threads(num_threads)
    {
#pragma omp for simd schedule(static)
        for (size_t i = 0; i < size; ++i) {
            float x = input_data[i];

            float x_cubed = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEF * x_cubed);

            float exp_val = std::exp(2.0f * inner);
            float tanh_approx = 1.0f - 2.0f / (exp_val + 1.0f);

            output_data[i] = HALF * x * (1.0f + tanh_approx);
        }
    }

    return output;
}