#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> result(size);
    
    constexpr float sqrt_2_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    constexpr float half = 0.5f;
    constexpr float two = 2.0f;
    
    #pragma omp parallel
    {
        const int num_threads = omp_get_num_threads();
        const int thread_id = omp_get_thread_num();
        const size_t chunk_size = (size + num_threads - 1) / num_threads;
        const size_t start = thread_id * chunk_size;
        const size_t end = (start + chunk_size < size) ? start + chunk_size : size;
        
        size_t i = start;
        for (; i + 3 < end; i += 4) {
            {
                float x = input[i];
                float x2 = x * x;
                float x3 = x2 * x;
                float inner = sqrt_2_pi * (x + coeff * x3);
                float exp_2z = std::exp(two * inner);
                result[i] = half * x * (1.0f + (exp_2z - 1.0f) / (exp_2z + 1.0f));
            }
            
            {
                float x = input[i + 1];
                float x2 = x * x;
                float x3 = x2 * x;
                float inner = sqrt_2_pi * (x + coeff * x3);
                float exp_2z = std::exp(two * inner);
                result[i + 1] = half * x * (1.0f + (exp_2z - 1.0f) / (exp_2z + 1.0f));
            }
            
            {
                float x = input[i + 2];
                float x2 = x * x;
                float x3 = x2 * x;
                float inner = sqrt_2_pi * (x + coeff * x3);
                float exp_2z = std::exp(two * inner);
                result[i + 2] = half * x * (1.0f + (exp_2z - 1.0f) / (exp_2z + 1.0f));
            }
            
            {
                float x = input[i + 3];
                float x2 = x * x;
                float x3 = x2 * x;
                float inner = sqrt_2_pi * (x + coeff * x3);
                float exp_2z = std::exp(two * inner);
                result[i + 3] = half * x * (1.0f + (exp_2z - 1.0f) / (exp_2z + 1.0f));
            }
        }
        
        for (; i < end; ++i) {
            float x = input[i];
            float x2 = x * x;
            float x3 = x2 * x;
            float inner = sqrt_2_pi * (x + coeff * x3);
            float exp_2z = std::exp(two * inner);
            result[i] = half * x * (1.0f + (exp_2z - 1.0f) / (exp_2z + 1.0f));
        }
    }
    
    return result;
}

