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
    if (input.empty() || batch <= 0) {
        return {};
    }

    int total_floats = static_cast<int>(input.size());
    if (total_floats % (2 * batch) != 0) {
        throw std::invalid_argument("Input size must be divisible by 2*batch");
    }
    int n = total_floats / (2 * batch);

    cufftHandle plan;
    cufftResult stat = cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    if (stat != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT plan creation failed");
    }

    size_t bytes = total_floats * sizeof(float);
    cufftComplex* d_data = nullptr;

    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    stat = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    if (stat != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("cuFFT forward transform failed");
    }

    stat = cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    if (stat != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("cuFFT inverse transform failed");
    }

    int total_complex = n * batch;
    const int block_size = 256;
    int grid_size = (total_complex + block_size - 1) / block_size;
    normalize_kernel << <grid_size, block_size >> > (d_data, n, total_complex);
    cudaDeviceSynchronize();

    std::vector<float> output(total_floats);
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}