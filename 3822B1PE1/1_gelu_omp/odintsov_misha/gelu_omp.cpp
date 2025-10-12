#include "gelu_omp.h"
#include <cmath>
#include <numbers>
#include <vector>

constexpr float M_PI = 3.14159265358979323846f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t len = input.size();
    std::vector<float> tmp(len);
    
    #pragma omp parallel for
    for (size_t i = 0; i < len; i++) {
        float x = input[i];
        tmp[i] = 0.5f * x * (1.0f + tanh(
            sqrt(2.0f / M_PI) *
            (x + 0.044715f * pow(x, 3))
        ));
    }
    return tmp;
}