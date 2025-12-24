#include "gelu_omp.h"

#include <omp.h>

static constexpr float myPow(float base, int degree)
{
	float answer = base;
	float finalMultiplicator = 1;
	while (degree > 1)
	{
		if (degree % 2 == 0)
		{
			answer *= answer;
			degree = degree >> 1;
		}
		else
		{
			finalMultiplicator *= answer;
			--degree;
		}
	}
	return answer * finalMultiplicator;
}

std::vector<float> GeluOMP(const std::vector<float>& input)
{
	std::vector<float> answer(input.size());
#pragma omp parallel
	{
#pragma omp for
		for (int index = 0; index < static_cast<int>(input.size()); ++index)
		{
			answer[index] = 0.5 * input[index] * (1  + std::tanh(constants::sqrt * (input[index] + constants::multiplicator * myPow(input[index], 3))));
		}
	}
	return answer;
}