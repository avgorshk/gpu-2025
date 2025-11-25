#include "fft_cufft.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <vector>
#include <stdexcept>
#include <string>
#include <cstring> 

inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + msg + " : " + cudaGetErrorString(e));
    }
}
inline void checkCuFFT(cufftResult r, const char* msg) {
    if (r != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("cuFFT error: ") + msg + " : " + std::to_string(r));
    }
}

__global__ void normalize_kernel(cufftComplex* data, float scale, size_t total) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (batch <= 0) throw std::invalid_argument("batch must be > 0");

    const size_t total_floats = input.size();
    if (total_floats == 0) return {};

    if (total_floats % 2 != 0) throw std::invalid_argument("input size must be even (pairs of floats)");
    if ((total_floats / 2) % static_cast<size_t>(batch) != 0)
        throw std::invalid_argument("input size inconsistent with batch: total_complex = input.size()/2 must be divisible by batch");

    const size_t total_complex = (total_floats / 2);
    const int n = static_cast<int>(total_complex / static_cast<size_t>(batch));

    if (n <= 0) throw std::invalid_argument("computed FFT length n <= 0");

    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(batch);
    const size_t bytes = elems * sizeof(cufftComplex);

    cufftHandle plan = 0;
    cudaStream_t stream = 0;
    cufftComplex* d_data = nullptr;
    cufftComplex* h_pinned = nullptr;

    try {
        checkCuda(cudaMallocHost((void**)&h_pinned, bytes), "cudaMallocHost for pinned host buffer");
        std::memcpy(h_pinned, input.data(), bytes);

        checkCuda(cudaMalloc((void**)&d_data, bytes), "cudaMalloc d_data");

        checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        checkCuFFT(cufftPlan1d(&plan, n, CUFFT_C2C, batch), "cufftPlan1d");
        checkCuFFT(cufftSetStream(plan, stream), "cufftSetStream");

        checkCuda(cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");

        checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "cufftExecC2C FORWARD");
        checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE), "cufftExecC2C INVERSE");

        const float inv_n = 1.0f / static_cast<float>(n);
        const int block = 256;
        const int grid = static_cast<int>((elems + block - 1) / block);
        normalize_kernel<<<grid, block, 0, stream>>>(d_data, inv_n, elems);

        checkCuda(cudaMemcpyAsync(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");

        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        std::vector<float> result(total_floats);
        std::memcpy(result.data(), h_pinned, bytes);

        cufftDestroy(plan);
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        cudaFreeHost(h_pinned);

        return result;
    } catch (...) {
        if (plan) cufftDestroy(plan);
        if (stream) cudaStreamDestroy(stream);
        if (d_data) cudaFree(d_data);
        if (h_pinned) cudaFreeHost(h_pinned);
        throw;
    }
}
