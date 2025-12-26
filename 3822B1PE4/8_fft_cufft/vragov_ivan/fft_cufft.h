#ifndef __FFT_CUFFT_H
#define __FFT_CUFFT_H

#include <vector>

// Input and Output format:
// A vector of floats representing complex numbers (Real, Imaginary)
// interleaved. Size of vector = 2 * n * batch
std::vector<float> FffCUFFT(const std::vector<float> &input, int batch);

#endif // __FFT_CUFFT_H
