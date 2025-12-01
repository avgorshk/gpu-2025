#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>
#include <cmath>

namespace constants
{
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;  // √(2/π)
    constexpr float gelu_coefficient = 0.044715f;
}

std::vector<float> GeluOMP(const std::vector<float>& input);

#endif // __GELU_OMP_H