#ifndef __GELU_OCL_H
#define __GELU_OCL_H

#include <vector>
#include <CL/cl.h>

std::vector<float> GeluOCL(const std::vector<float>& input);

#endif // __GELU_OCL_H