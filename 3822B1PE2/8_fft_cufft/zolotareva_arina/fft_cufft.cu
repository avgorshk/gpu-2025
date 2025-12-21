#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

__global__ void normalize_kernel(cufftComplex* data, int n_total, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_total) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    const int total_complex = static_cast<int>(input.size() / 2);
    const int n = total_complex / batch;

    std::vector<float> output(input.size());

    cufftComplex* d_data = nullptr;
    size_t bytes = total_complex * sizeof(cufftComplex);

    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int threads = 256;
    int blocks = (total_complex + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(d_data, total_complex, n);

    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}
