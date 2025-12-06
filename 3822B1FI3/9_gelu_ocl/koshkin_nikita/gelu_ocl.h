#ifndef __GELU_OCL_H
#define __GELU_OCL_H

#include <vector>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform);

#endif // __GELU_OCL_H