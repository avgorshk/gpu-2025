#ifndef __FFT_CUFFT_H
#define __FFT_CUFFT_H

#include <vector>

std::vector<float> FftCUFFT(const std::vector<float>& input, int batch);

#endif // __FFT_CUFFT_H