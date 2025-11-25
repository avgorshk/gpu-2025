#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstring>   
#include <cmath>     
#include <string>

inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

__global__ void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= n) return;

    const float x = in[idx];
    const float x3 = x * x * x;

    constexpr float k = 0.7978845608028654f;
    constexpr float coeff = 0.044715f;

    float inner = k * fmaf(coeff, x3, x); 

    float tanh_val;
    if (inner >= 0.0f) {
        float z = __expf(-2.0f * inner);
        tanh_val = (1.0f - z) / (1.0f + z);
    } else {
        float z = __expf(2.0f * inner);
        tanh_val = (z - 1.0f) / (z + 1.0f);
    }

    out[idx] = 0.5f * x * (1.0f + tanh_val);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) return {};

    std::vector<float> result(n);

    float *d_in = nullptr, *d_out = nullptr;
    float *pinned_in = nullptr, *pinned_out = nullptr;
    cudaStream_t stream = nullptr;

    try {
        const size_t bytes = n * sizeof(float);

        cudaCheck(cudaMalloc((void**)&d_in, bytes), "cudaMalloc d_in");
        cudaCheck(cudaMalloc((void**)&d_out, bytes), "cudaMalloc d_out");

        cudaCheck(cudaMallocHost((void**)&pinned_in, bytes), "cudaMallocHost pinned_in");
        cudaCheck(cudaMallocHost((void**)&pinned_out, bytes), "cudaMallocHost pinned_out");

        std::memcpy(pinned_in, input.data(), bytes);

        cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate");

        cudaCheck(cudaMemcpyAsync(d_in, pinned_in, bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");

        int blockSize = 256;
        int minGridSize = 0;
        cudaError_t occ_err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)gelu_kernel, 0, 0);
        if (occ_err != cudaSuccess) {
            blockSize = 256;
        }
        if (blockSize <= 0) blockSize = 256;
        if (blockSize > 1024) blockSize = 1024;

        const int grid = static_cast<int>((n + blockSize - 1) / blockSize);

        gelu_kernel<<<grid, blockSize, 0, stream>>>(d_in, d_out, n);

        cudaCheck(cudaGetLastError(), "Kernel launch");

        cudaCheck(cudaMemcpyAsync(pinned_out, d_out, bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");

        cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        std::memcpy(result.data(), pinned_out, bytes);

        cudaCheck(cudaStreamDestroy(stream), "cudaStreamDestroy");
        cudaCheck(cudaFreeHost(pinned_in), "cudaFreeHost pinned_in");
        cudaCheck(cudaFreeHost(pinned_out), "cudaFreeHost pinned_out");
        cudaCheck(cudaFree(d_in), "cudaFree d_in");
        cudaCheck(cudaFree(d_out), "cudaFree d_out");

        return result;
    } catch (...) {
        if (stream) cudaStreamDestroy(stream);
        if (pinned_in) cudaFreeHost(pinned_in);
        if (pinned_out) cudaFreeHost(pinned_out);
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        throw;
    }
}
