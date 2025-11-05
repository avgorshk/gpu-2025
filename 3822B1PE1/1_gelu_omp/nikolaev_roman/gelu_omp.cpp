#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

constexpr float GELU_SCALE = 0.7978845608028654f;
constexpr float GELU_CF = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input) { 
  const size_t size = input.size();
  std::vector<float> result(size);

  #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += 4) {
            size_t remaining = size - i;
            size_t unroll_count = (remaining >= 4) ? 4 : remaining;
            
            for (size_t j = 0; j < unroll_count; ++j) {
                size_t idx = i + j;
                float x = input[idx];
                
                float x_cubed = x * x * x;
                float inner = GELU_SCALE * (x + GELU_CF * x_cubed);
                
                float tanh_val = std::tanh(inner);
                
                result[idx] = 0.5f * x * (1.0f + tanh_val);
            }
        }
    }
    return result;
}