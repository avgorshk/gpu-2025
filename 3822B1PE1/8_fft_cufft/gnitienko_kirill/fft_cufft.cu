#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void normalize_kernel(cufftComplex* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    int total_elements = n * batch;
    std::vector<float> output(input.size());

    cufftHandle plan;
    cufftComplex* d_data;
    size_t complex_data_size = total_elements * sizeof(cufftComplex);

    cudaMalloc(&d_data, complex_data_size);
    cudaMemcpy(d_data, input.data(), complex_data_size, cudaMemcpyHostToDevice);

    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    normalize_kernel<<<(total_elements + 255) / 256, 256>>>(d_data, total_elements, 1.0f / n);

    cudaMemcpy(output.data(), d_data, complex_data_size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}