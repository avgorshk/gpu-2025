#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <omp.h>


std::vector<float> GeluOMP(const std::vector<float>& input) {

    std::vector<float> result(input.size());
    
    const float PI = 3.14159265358979323846f;
    const float COEFF = 2.0f / PI;
    const int size = static_cast<int>(input.size());

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float x3 = x * x * x;

        result[i] = 0.5f * x * (1.0f + tanhf(COEFF * (x + 0.044715f * x3)));
    }
    
    return result;
}