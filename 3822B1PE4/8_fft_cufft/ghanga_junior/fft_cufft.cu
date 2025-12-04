#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <vector>

__global__ void normalize_kernel(cufftComplex* data, int total, float inv_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        data[i].x *= inv_n;
        data[i].y *= inv_n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    // input contient des paires (real, imag) pour chaque point
    // taille attendue : 2 * n * batch
    if (batch <= 0 || input.empty() || input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("Invalid input parameters");
    }

    int n = static_cast<int>(input.size() / (2 * batch));
    int total_complex = n * batch;
    size_t bytes = sizeof(cufftComplex) * total_complex;

    // d_data va contenir total_complex éléments complexes (réel + imaginaire)
    cufftComplex* d_data = nullptr;
    if (cudaMalloc(&d_data, bytes) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }

    // On copie directement les floats (real, imag, real, imag, ...) dans d_data
    if (cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_data);
        throw std::runtime_error("cudaMemcpy (host->device) failed");
    }

    // Plan cuFFT pour FFT complexe -> complexe
    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        cudaFree(d_data);
        throw std::runtime_error("cufftPlan1d failed");
    }

    // Forward FFT
    if (cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("Forward FFT failed");
    }

    // Inverse FFT
    if (cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("Inverse FFT failed");
    }

    // Normalisation par n sur le GPU
    float inv_n = 1.0f / static_cast<float>(n);
    int blockSize = 256;
    int gridSize  = (total_complex + blockSize - 1) / blockSize;
    normalize_kernel<<<gridSize, blockSize>>>(d_data, total_complex, inv_n);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("Kernel execution failed");
    }

    // Récupérer les données en float interleavé (real, imag, ...)
    std::vector<float> output(input.size());
    if (cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("cudaMemcpy (device->host) failed");
    }

    cufftDestroy(plan);
    cudaFree(d_data);
    return output;
}
