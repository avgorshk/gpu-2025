#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void normalize_kernel(cufftComplex* data, int n, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float scale = 1.0f / static_cast<float>(n);
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    size_t total_size = input.size();
    if (total_size % (2 * batch) != 0) {
        throw std::invalid_argument("Invalid Input size");
    }
    int n = static_cast<int>(total_size / (2 * batch));

    cufftComplex* d_data = nullptr;
    size_t complex_size = n * batch * sizeof(cufftComplex);

    cudaMalloc(&d_data, complex_size);
    cudaMemcpy(d_data, input.data(), complex_size, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int total_elements = n * batch;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    normalize_kernel<<<grid_size, block_size>>>(d_data, n, total_elements);

    std::vector<float> result(total_size);
    cudaMemcpy(result.data(), d_data, complex_size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return result;
}