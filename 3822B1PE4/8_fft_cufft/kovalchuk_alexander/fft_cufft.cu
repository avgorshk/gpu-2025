#include "fft_cufft.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>

__global__ void normalize_kernel(cufftComplex* data, int n, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * batch;
    if (idx < total) {
        float scale = 1.0f / static_cast<float>(n);
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (batch <= 0) {
        throw std::runtime_error("FffCUFFT: batch must be > 0");
    }

    const std::size_t total_floats = input.size();
    if (total_floats == 0 || total_floats % (2 * batch) != 0) {
        throw std::runtime_error("FffCUFFT: wrong input size");
    }

    int n = static_cast<int>(total_floats / (2 * batch));
    std::size_t total_complex = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);

    std::vector<float> output(total_floats);

    cufftComplex* d_data = nullptr;
    cudaError_t cuda_status;
    cufftResult cufft_status;
    cufftHandle plan;

    cuda_status = cudaMalloc(&d_data, total_complex * sizeof(cufftComplex));
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("cudaMalloc d_data failed");
    }

    cuda_status = cudaMemcpy(d_data,
                             reinterpret_cast<const cufftComplex*>(input.data()),
                             total_complex * sizeof(cufftComplex),
                             cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_data);
        throw std::runtime_error("cudaMemcpy H2D failed");
    }

    cufft_status = cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    if (cufft_status != CUFFT_SUCCESS) {
        cudaFree(d_data);
        throw std::runtime_error("cufftPlan1d failed");
    }

    cufft_status = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    if (cufft_status != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("cufftExecC2C forward failed");
    }

    cufft_status = cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    if (cufft_status != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("cufftExecC2C inverse failed");
    }

    int threads = 256;
    int blocks = static_cast<int>((total_complex + threads - 1) / threads);
    normalize_kernel<<<blocks, threads>>>(d_data, n, batch);

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("cudaDeviceSynchronize failed");
    }

    cuda_status = cudaMemcpy(reinterpret_cast<cufftComplex*>(output.data()),
                             d_data,
                             total_complex * sizeof(cufftComplex),
                             cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("cudaMemcpy D2H failed");
    }

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}
