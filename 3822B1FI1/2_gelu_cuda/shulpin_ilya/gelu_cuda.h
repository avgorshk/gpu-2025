#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> GeluCUDA(const std::vector<float>& input);

#endif // __GELU_CUDA_H