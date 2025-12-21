#include "fft_cufft.h"
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void NormalizeKernel(float *data, size_t n, float factor) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        data[index] /= factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch) {
    if (input.empty() || batch <= 0) {
        return {};
    }

    size_t size = input.size();
    size_t data_size = size * sizeof(float);
    size_t block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;
    auto signal_length = static_cast<int>(size / (2 * batch));

    float *d_data = nullptr;
    cudaMalloc(&d_data, data_size);

    cudaMemcpy(d_data, input.data(), data_size, cudaMemcpyHostToDevice);
    cufftHandle plan;
    cufftPlan1d(&plan, signal_length, CUFFT_C2C, batch);
    cufftExecC2C(plan, (cufftComplex *) d_data, (cufftComplex *) d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex *) d_data, (cufftComplex *) d_data, CUFFT_INVERSE);
    NormalizeKernel<<<num_blocks, block_size>>>(d_data, size, static_cast<float>(signal_length));
    std::vector<float> output(size);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_data, data_size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);
    return output;
}
