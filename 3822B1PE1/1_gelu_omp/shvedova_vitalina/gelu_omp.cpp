#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& in) {
    std::vector<float> out(in.size());
    static const float c = std::sqrt(2.0f / static_cast<float>(M_PI));

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(in.size()); i++) {
        float x = in[i];
        float inner = c * (x + 0.044715f * x * x * x);

        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
    return out;
}
