#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <cassert>

__global__ void normalizeFFT(cufftComplex* data, float scale, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    assert(batch > 0);
    int n = static_cast<int>(input.size() / (2 * batch));
    int total = n * batch;

    size_t bytes = total * sizeof(cufftComplex);
    cufftComplex* d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpyAsync(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int threads = 512;
    int blocks = (total + threads - 1) / threads;
    normalizeFFT<<<blocks, threads>>>(d_data, 1.0f / static_cast<float>(n), total);

    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);
    return output;
}
