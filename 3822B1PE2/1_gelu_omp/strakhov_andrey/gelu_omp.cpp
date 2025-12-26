#include "gelu_omp.h"
#define M_PI 3.14159265358979323846
std::vector<float> GeluOMP(const std::vector<float> &input)
{
    // Place your implementation here
    std::vector<float> res(input.size());
    for (int i = 0; i < input.size(); i++)
    {
        float x = input[i];
        res[i] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI * (x + 0.044715f * x * x * x))));
    }
    return res;
    // gelu(x) = 0.5*x*(1 + tanh(sqrt(2/M_PI*(x+0.044715*x*x*x)))
    // first one: 0.5*x*(1+(1-(2*(1/(1+exp(sqrt(2/PI*(x + 0.0044715*x*x*x))*2))))))
    //  2/(1+exp(-2*x))-1
    // 0.5*x*(2/(1+exp(-2*sqrt(2/M_PI*(x+0.044715*x*x*x))))
}