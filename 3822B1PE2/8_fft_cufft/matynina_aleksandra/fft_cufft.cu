#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

__global__ void normalize_kernel(cufftComplex* data, int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= factor;
        data[idx].y *= factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (batch <= 0) {
        throw std::invalid_argument("batch must be positive");
    }
    
    if (input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("input size must be divisible by 2 * batch");
    }
    
    int n = input.size() / (2 * batch);
    int total_complex = n * batch;
    
    cufftComplex* d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, total_complex * sizeof(cufftComplex));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
    }
    
    cufftComplex* h_data = new cufftComplex[total_complex];
    for (int i = 0; i < total_complex; i++) {
        h_data[i].x = input[2 * i];
        h_data[i].y = input[2 * i + 1];
    }
    err = cudaMemcpy(d_data, h_data, total_complex * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        delete[] h_data;
        cudaFree(d_data);
        throw std::runtime_error("Failed to copy data to device");
    }
    delete[] h_data;
    
    cufftHandle plan_forward;
    cufftResult cufft_err = cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch);
    if (cufft_err != CUFFT_SUCCESS) {
        cudaFree(d_data);
        throw std::runtime_error("Failed to create forward FFT plan");
    }
    
    cufft_err = cufftExecC2C(plan_forward, d_data, d_data, CUFFT_FORWARD);
    if (cufft_err != CUFFT_SUCCESS) {
        cufftDestroy(plan_forward);
        cudaFree(d_data);
        throw std::runtime_error("Failed to execute forward FFT");
    }
    
    cufftHandle plan_inverse;
    cufft_err = cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch);
    if (cufft_err != CUFFT_SUCCESS) {
        cufftDestroy(plan_forward);
        cudaFree(d_data);
        throw std::runtime_error("Failed to create inverse FFT plan");
    }
    
    cufft_err = cufftExecC2C(plan_inverse, d_data, d_data, CUFFT_INVERSE);
    if (cufft_err != CUFFT_SUCCESS) {
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        cudaFree(d_data);
        throw std::runtime_error("Failed to execute inverse FFT");
    }
    
    float normalization_factor = 1.0f / static_cast<float>(n);
    int num_blocks = (total_complex + 255) / 256;
    normalize_kernel<<<num_blocks, 256>>>(d_data, total_complex, normalization_factor);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        cudaFree(d_data);
        throw std::runtime_error("Failed to execute normalization kernel");
    }
    cudaDeviceSynchronize();
    
    cufftComplex* h_result = new cufftComplex[total_complex];
    err = cudaMemcpy(h_result, d_data, total_complex * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        delete[] h_result;
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        cudaFree(d_data);
        throw std::runtime_error("Failed to copy data from device");
    }
    
    std::vector<float> result(2 * total_complex);
    for (int i = 0; i < total_complex; i++) {
        result[2 * i] = h_result[i].x;
        result[2 * i + 1] = h_result[i].y;
    }
    delete[] h_result;
    
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cudaFree(d_data);
    
    return result;
}

