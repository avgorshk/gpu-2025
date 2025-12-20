#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void normalizeKernel(cufftComplex* data, int total, float inv_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x *= inv_n;
        data[idx].y *= inv_n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    std::vector<float> output(input.size(), 0.0f);
    if (batch <= 0 || input.empty()) return output;
    if ((input.size() & 1u) != 0u) return output;

    int total_complex = static_cast<int>(input.size() / 2);
    if (total_complex % batch != 0) return output;

    int n = total_complex / batch;
    if (n <= 0) return output;

    cufftComplex* d_data = nullptr;
    size_t bytes = static_cast<size_t>(total_complex) * sizeof(cufftComplex);

    if (cudaMalloc(&d_data, bytes) != cudaSuccess) return output;

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        cudaFree(d_data);
        return output;
    }

    if (cudaMemcpyAsync(d_data, input.data(), bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return output;
    }

    cufftHandle plan;
    int rank = 1;
    int dims[1] = { n };
    int inembed[1] = { n };
    int onembed[1] = { n };

    if (cufftPlanMany(&plan, rank, dims,
                     inembed, 1, n,
                     onembed, 1, n,
                     CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return output;
    }

    if (cufftSetStream(plan, stream) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return output;
    }

    if (cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) != CUFFT_SUCCESS ||
        cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return output;
    }

    int threads = 256;
    int blocks = (total_complex + threads - 1) / threads;
    float inv_n = 1.0f / static_cast<float>(n);
    normalizeKernel<<<blocks, threads, 0, stream>>>(d_data, total_complex, inv_n);

    if (cudaGetLastError() != cudaSuccess) {
        cufftDestroy(plan);
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return output;
    }

    if (cudaMemcpyAsync(output.data(), d_data, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        cufftDestroy(plan);
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return std::vector<float>(input.size(), 0.0f);
    }

    cudaStreamSynchronize(stream);

    cufftDestroy(plan);
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    return output;
}
