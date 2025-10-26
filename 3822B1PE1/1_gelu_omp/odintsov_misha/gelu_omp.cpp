#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <omp.h>



std::vector<float> GeluOMP(const std::vector<float>& input) {
    const float My_PI = 3.14159265358979323846f;
    size_t len = input.size();
    std::vector<float> tmp(len);
    const float sq = sqrt(2.0f / My_PI);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < len; i++) {
        float x = input[i];
        tmp[i] = 0.5f * x * (1.0f + tanh(
            sq *
            (x + 0.044715f * x*x*x)
        ));
    }
    return tmp;
}