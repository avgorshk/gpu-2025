#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#include <cuda_runtime.h>
#include <math_constants.h>

#include <vector>

std::vector<float> GeluCUDA(const std::vector<float>& input);

#endif  // __GELU_CUDA_H