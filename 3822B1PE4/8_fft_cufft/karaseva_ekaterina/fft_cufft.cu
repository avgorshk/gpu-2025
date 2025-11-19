#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>

__global__ void normalize_kernel(cufftComplex* data, int total, float inv_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        data[i].x *= inv_n;
        data[i].y *= inv_n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (batch <= 0 || input.empty() || input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("Invalid input parameters");
    }

    int n = input.size() / (2 * batch);
    int total_complex = n * batch;
    size_t bytes = sizeof(cufftComplex) * total_complex;

    cufftComplex* d_data;
    if (cudaMalloc(&d_data, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    if (cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_data);
        throw std::runtime_error("cudaMemcpy failed");
    }

    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        cudaFree(d_data);
        throw std::runtime_error("cufftPlan1d failed");
    }

    if (cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("Forward FFT failed");
    }

    if (cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("Inverse FFT failed");
    }

    float inv_n = 1.0f / n;
    normalize_kernel<<<(total_complex + 255) / 256, 256>>>(d_data, total_complex, inv_n);
    
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("Kernel execution failed");
    }

    std::vector<float> output(input.size());
    if (cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("cudaMemcpy failed");
    }

    cufftDestroy(plan);
    cudaFree(d_data);
    return output;
}