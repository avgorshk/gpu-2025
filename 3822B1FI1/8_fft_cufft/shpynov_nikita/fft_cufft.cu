#include "fft_cufft.h"

#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUFFT(call) \
    if ((call) != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

std::vector<float> FffCUFFT(
    const std::vector<float>& input,
    int batch)
{
    const int n = input.size() / (2 * batch);
    const size_t complex_count = n * batch;

    cufftComplex* d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, complex_count * sizeof(cufftComplex)));

    CHECK_CUDA(cudaMemcpy(
        d_data,
        input.data(),
        complex_count * sizeof(cufftComplex),
        cudaMemcpyHostToDevice
    ));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(
        &plan,
        n,
        CUFFT_C2C,
        batch
    ));

    CHECK_CUFFT(cufftExecC2C(
        plan,
        d_data,
        d_data,
        CUFFT_FORWARD
    ));

    CHECK_CUFFT(cufftExecC2C(
        plan,
        d_data,
        d_data,
        CUFFT_INVERSE
    ));

    std::vector<float> output(input.size());
    CHECK_CUDA(cudaMemcpy(
        output.data(),
        d_data,
        complex_count * sizeof(cufftComplex),
        cudaMemcpyDeviceToHost
    ));

    const float scale = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] *= scale;
    }

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}