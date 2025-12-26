#include "gelu_omp.h"
#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
	std::vector<float> output(input.size());
	const size_t n = input.size();

	constexpr float sqrt_2_over_pi = 0.7978845608028654f;
	constexpr float coef = 0.044715f;

#pragma omp parallel for
	for (size_t i = 0; i < n; ++i) {
		float x = input[i];
		float x3 = x * x * x;
		float inner = sqrt_2_over_pi * (x + coef * x3);

		float exp_val = std::exp(2.0f * inner);
		float tanh_approx = 1.0f - 2.0f / (exp_val + 1.0f);

		output[i] = 0.5f * x * (1.0f + tanh_approx);
	}

	return output;
}