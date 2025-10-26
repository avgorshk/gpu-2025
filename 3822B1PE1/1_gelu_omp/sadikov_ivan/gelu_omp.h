#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>
#include <cmath>

namespace constants
{
	constexpr float sqrt = 0.7978;
	constexpr float multiplicator = 0.044715;
}

static constexpr float myPow(float base, int degree);

std::vector<float> GeluOMP(const std::vector<float>& input);

#endif // __GELU_OMP_H