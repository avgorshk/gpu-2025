#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<float> output(n);

    // sqrt(2/pi) en float
    const float c = 0.7978845608f;    // ≈ std::sqrt(2.0f / M_PI)
    const float alpha = 0.044715f;

    // Parallelisation + vectorisation
    #pragma omp parallel for simd
    for (int i = 0; i < n; ++i) {
        float x  = input[i];
        float x2 = x * x;
        float x3 = x2 * x;

        float v  = x + alpha * x3;
        float z  = c * v;

        // Approximation rapide de tanh(z)
        float e  = std::exp(-2.0f * z);     // e^{-2z}
        float t  = (1.0f - e) / (1.0f + e); // tanh(z)

        output[i] = 0.5f * x * (1.0f + t);
    }

    return output;
}
