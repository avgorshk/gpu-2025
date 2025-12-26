#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> result(size);
    
    if (size == 0) {
        return result;
    }

    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;

    const float* inputPtr = input.data();
    float* resultPtr = result.data();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float x = inputPtr[i];
        float xCubed = x * x * x;
        float inner = kSqrt2OverPi * (x + kCoeff * xCubed);
        float tanhVal = std::tanh(inner);
        resultPtr[i] = 0.5f * x * (1.0f + tanhVal);
    }

    return result;
}
