#ifndef __FFT_CUFFT_H
#define __FFT_CUFFT_H

#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch);

#endif // __FFT_CUFFT_H