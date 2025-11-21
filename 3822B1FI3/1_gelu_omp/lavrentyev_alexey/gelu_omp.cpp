#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

using std::vector;

vector<float> GeluOMP(const vector<float>& input) {
	
	vector<float> ans(input.size());
	#pragma omp parallel for
	for (std::size_t index = 0; index < input.size(); ++index) {
		float var = input[index];
        float cube = var * var * var;
		ans[index] = 0.5f * var * (1.f + std::tanh(0.79788f * (var + 0.044715f * cube)));
	}
	return ans;
}
