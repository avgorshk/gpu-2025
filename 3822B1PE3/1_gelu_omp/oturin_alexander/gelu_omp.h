#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <cmath>
#include <numbers>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input);

#endif  // __GELU_OMP_H
