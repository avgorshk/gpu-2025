#include "gelu_omp.h"

#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const float kAlpha = 0.7978845608f; // sqrt(2/pi)
    const float kBeta = 0.044715f;
    const int n = static_cast<int>(input.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float x = input[i];
        float x2 = x * x;
        float x3 = x2 * x;
        float u = kAlpha * (x + kBeta * x3);
        float t = tanhf(u);
        output[i] = 0.5f * x * (1.0f + t);
    }

    return output;
}
