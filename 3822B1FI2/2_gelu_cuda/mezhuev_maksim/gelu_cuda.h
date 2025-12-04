#ifndef GELU_CUDA_IMPLEMENTATION_H
#define GELU_CUDA_IMPLEMENTATION_H

#include <vector>

std::vector<float> runGeluOnGPU(const std::vector<float>& source);

#endif