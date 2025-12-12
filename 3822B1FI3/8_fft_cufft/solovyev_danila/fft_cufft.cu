
#include "fft_cufft.h"

#include <vector>
#include <stdexcept>
#include <cufft.h>

__global__ void normalizeKernel(cufftComplex* data, int count, float norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx].x /= norm;
        data[idx].y /= norm;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {

    int total_complex = input.size() / 2;
    int n = total_complex / batch;

    size_t total_size_bytes = sizeof(cufftComplex) * total_complex;

    cufftComplex* d_data = nullptr;
    cudaMalloc(&d_data, total_size_bytes);
    cudaMemcpy(d_data, reinterpret_cast<const cufftComplex*>(input.data()), total_size_bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlanMany(
            &plan,
            1,
            &n,
            nullptr, 1, n,
            nullptr, 1, n,
            CUFFT_C2C,
            batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int threads = 256;
    int blocks = (total_complex + threads - 1) / threads;

    normalizeKernel<<<blocks, threads>>>(d_data, total_complex, static_cast<float>(n));
    cudaDeviceSynchronize();
    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), reinterpret_cast<float*>(d_data), total_size_bytes, cudaMemcpyDeviceToHost);
    cufftDestroy(plan);
    cudaFree(d_data);
    return output;
}
