#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>
#include <numbers>
#include <cmath>

namespace constants
{
	const float sqrt = std::sqrt((2 / std::numbers::pi));
	constexpr float multiplicator = 0.044715;
}

static constexpr float myPow(float base, int degree);

std::vector<float> GeluOMP(const std::vector<float>& input);

#endif // __GELU_OMP_H