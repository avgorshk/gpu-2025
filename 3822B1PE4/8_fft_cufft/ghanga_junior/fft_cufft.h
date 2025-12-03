#ifndef __FFT_CUFFT_H
#define __FFT_CUFFT_H

#include <vector>

std::vector<float> FftCuFFT(const std::vector<float>& input, int n);

#endif // __FFT_CUFFT_H
