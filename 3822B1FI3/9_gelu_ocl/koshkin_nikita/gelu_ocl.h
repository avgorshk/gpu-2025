#ifndef __GELU_OCL_H
#define __GELU_OCL_H

#include <CL/cl.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform);

#endif // __GELU_OCL_H