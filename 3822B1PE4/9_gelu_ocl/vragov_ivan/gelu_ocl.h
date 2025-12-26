#ifndef __GELU_OCL_H
#define __GELU_OCL_H

#include <vector>

// Platform index to select (usually 0 is NVIDIA on systems with CUDA installed,
// but depends on setup)
std::vector<float> GeluOCL(const std::vector<float> &input, int platform_id);

#endif // __GELU_OCL_H
