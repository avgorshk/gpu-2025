#ifndef __GELU_OCL_H
#define __GELU_OCL_H

#define CL_TARGET_OPENCL_VERSION 120
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <CL/cl.h>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform);

#endif // __GELU_OCL_H