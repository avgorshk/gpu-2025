#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

__global__ void normalize_kernel(cufftComplex* data, int n, float norm) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx].x *= norm;
        data[idx].y *= norm;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int totalSize = input.size();
    int n = totalSize / (2 * batch);
    int complex_size = n * batch;

    cufftComplex* d_data;
    cudaMalloc(&d_data, complex_size * sizeof(cufftComplex));

    cudaMemcpy(d_data, input.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int threads = 256;
    int blocks = (complex_size + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(d_data, complex_size, 1.0f / n);
    cudaDeviceSynchronize();

    std::vector<float> output(totalSize);
    cudaMemcpy(output.data(), d_data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}