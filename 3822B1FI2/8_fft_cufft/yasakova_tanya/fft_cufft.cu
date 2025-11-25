#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>


__global__ void NormalizeFFT(cufftComplex* data, int count, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batchCount) {
    int totalElems = input.size();
    int fftLength = totalElems / (2 * batchCount);
    int numComplex = totalElems / 2;

    if (totalElems % (2 * batchCount) != 0 || fftLength == 0) {
        throw std::runtime_error("Invalid input size or batch count");
    }

    cufftComplex* d_signal;
    cudaMalloc(&d_signal, numComplex * sizeof(cufftComplex));
    cudaMemcpy(d_signal, input.data(), totalElems * sizeof(float), cudaMemcpyHostToDevice);

    cufftHandle fftPlan;
    cufftPlan1d(&fftPlan, fftLength, CUFFT_C2C, batchCount);

    cufftExecC2C(fftPlan, d_signal, d_signal, CUFFT_FORWARD);

    cufftExecC2C(fftPlan, d_signal, d_signal, CUFFT_INVERSE);

    float scale = 1.0f / fftLength;
    int threads = 256;
    int blocks = (numComplex + threads - 1) / threads;
    NormalizeFFT<<<blocks, threads>>>(d_signal, numComplex, scale);

    std::vector<float> output(totalElems);
    cudaMemcpy(output.data(), d_signal, totalElems * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_signal);
    cufftDestroy(fftPlan);

    return output;
}