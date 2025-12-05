#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void norm(cufftComplex* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    const int n = input.size() / (2 * batch);
    const int total = n * batch;
    const size_t bytes = total * sizeof(cufftComplex);

    std::vector<float> output(input.size());

    cufftComplex *d_in, *d_fft, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_fft, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_in, d_fft, CUFFT_FORWARD);
    cufftExecC2C(plan, d_fft, d_out, CUFFT_INVERSE);

    const int blockSize = 128;
    const int gridSize = (total + blockSize - 1) / blockSize;
    norm<<<gridSize, blockSize>>>(d_out, total, 1.0f / n);

    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_fft);
    cudaFree(d_out);

    return output;
}
