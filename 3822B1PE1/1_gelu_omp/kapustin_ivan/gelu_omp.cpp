#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
	const size_t n = input.size();
	std::vector<float> output(n);

	const float k = 0.7978845608028654f;
	const float c = 0.044715f;

#pragma omp parallel for
	for (long long i = 0; i < (long long)n; i++) {

		float x = input[i];
		float x3 = x * x * x;

		float v = k * (x + c * x3);
		float e = expf(2.0f * v);

		float t = (e - 1.0f) / (e + 1.0f);

		output[i] = 0.5f * x * (1.0f + t);
	}

	return output;
}