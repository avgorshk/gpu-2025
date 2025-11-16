#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cassert>
#include <vector>

__global__ void normalize_kernel(cufftComplex* data, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    assert(batch > 0);
    assert(input.size() % (2 * batch) == 0);

    int n = input.size() / (2 * batch);
    size_t total_elements = n * batch;
    size_t size = total_elements * sizeof(cufftComplex);

    cufftHandle plan;
    cufftResult cufftErr = cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    if (cufftErr != CUFFT_SUCCESS)
        throw std::runtime_error("cufftPlan1d failed");

    cufftComplex* d_data = nullptr;
    cudaError_t cudaErr = cudaMalloc(&d_data, size);
    if (cudaErr != cudaSuccess) {
        cufftDestroy(plan);
        throw std::runtime_error("cudaMalloc failed");
    }

    cudaErr = cudaMemcpy(d_data, input.data(), size, cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        cudaFree(d_data);
        cufftDestroy(plan);
        throw std::runtime_error("cudaMemcpy H2D failed");
    }

    cufftErr = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    if (cufftErr != CUFFT_SUCCESS)
        throw std::runtime_error("Forward FFT failed");

    cufftErr = cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    if (cufftErr != CUFFT_SUCCESS)
        throw std::runtime_error("Inverse FFT failed");

    float scale = 1.0f / n;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, scale, total_elements);

    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess)
        throw std::runtime_error("Kernel execution failed");

    std::vector<float> result(input.size());
    cudaErr = cudaMemcpy(result.data(), d_data, size, cudaMemcpyDeviceToHost);
    if (cudaErr != cudaSuccess)
        throw std::runtime_error("cudaMemcpy D2H failed");

    cudaFree(d_data);
    cufftDestroy(plan);

    return result;
}
