#include "fft_cufft.h"

#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);

    cufftComplex* d_data = nullptr;
    size_t bytes = input.size() * sizeof(float);

    cudaMalloc((void**)&d_data, bytes);
    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle handle;

    cufftPlan1d(&handle, n, CUFFT_C2C, batch);

    cufftExecC2C(handle, d_data, d_data, CUFFT_FORWARD);

    cufftExecC2C(handle, d_data, d_data, CUFFT_INVERSE);

    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= n;
    }

    cudaFree(d_data);
    cufftDestroy(handle);

    return output;
}
