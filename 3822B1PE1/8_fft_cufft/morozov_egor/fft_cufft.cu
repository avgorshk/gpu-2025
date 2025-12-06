#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string>
#include "fft_cufft.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::string err_msg = "CUDA Error: " + std::string(cudaGetErrorString(error)) + \
                                  " at " + __FILE__ + ":" + std::to_string(__LINE__); \
            throw std::runtime_error(err_msg); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult result = call; \
        if (result != CUFFT_SUCCESS) { \
            std::string err_msg = "cuFFT Error: " + std::to_string(result) + \
                                  " at " + __FILE__ + ":" + std::to_string(__LINE__); \
            throw std::runtime_error(err_msg); \
        } \
    } while(0)

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (batch <= 0) {
        throw std::invalid_argument("batch must be positive");
    }

    if (input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("input size must be divisible by 2*batch");
    }

    size_t total_complex_elements = input.size() / 2;
    size_t n = total_complex_elements / batch;

    if (n == 0) {
        throw std::invalid_argument("input too small for given batch size");
    }

    std::vector<float> output(input.size());

    if (input.empty()) {
        return output;
    }

    cufftComplex* d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_data, total_complex_elements * sizeof(cufftComplex)));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_C2C, batch));

    CUDA_CHECK(cudaMemcpy(d_data, input.data(),
                          total_complex_elements * sizeof(cufftComplex),
                          cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    CUDA_CHECK(cudaMemcpy(output.data(), d_data,
                          total_complex_elements * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));

    float scale = 1.0f / n;
    for (size_t i = 0; i < output.size(); i++) {
        output[i] *= scale;
    }

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_data));

    return output;
}