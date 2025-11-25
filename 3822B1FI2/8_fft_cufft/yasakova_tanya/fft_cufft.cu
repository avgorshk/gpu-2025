#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cmath>

__global__ void apply_scale(cufftComplex* arr, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx].x *= s;
        arr[idx].y *= s;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.empty()) return std::vector<float>();
    if (batch <= 0) throw std::invalid_argument("Batch not positive");
    if (input.size() % (2 * batch) != 0) throw std::invalid_argument("Input size not 2 * batch");
    
    int N = input.size() / (2 * batch);
    if (N <= 0) throw std::invalid_argument("Invalid length");

    size_t comp_count = N * batch;
    size_t mem_bytes = comp_count * sizeof(cufftComplex);

    std::vector<float> out_data(input.size());

    cufftComplex* d_in = nullptr;
    cufftComplex* d_buf = nullptr;
    cufftComplex* d_res = nullptr;

    cudaMalloc(&d_in, mem_bytes);
    cudaMalloc(&d_buf, mem_bytes);
    cudaMalloc(&d_res, mem_bytes);
    cudaMemcpy(d_in, input.data(), mem_bytes, cudaMemcpyHostToDevice);

    cufftHandle fft_plan, ifft_plan;
    cufftPlan1d(&fft_plan, N, CUFFT_C2C, batch);
    cufftPlan1d(&ifft_plan, N, CUFFT_C2C, batch);

    cufftExecC2C(fft_plan, d_in, d_buf, CUFFT_FORWARD);
    cufftExecC2C(ifft_plan, d_buf, d_res, CUFFT_INVERSE);

    float factor = 1.0f / N;
    int total_elems = comp_count;

    int block_size = 256;
    int grid_size = (total_elems + block_size - 1) / block_size;

    apply_scale<<<grid_size, block_size>>>(d_res, factor, total_elems);

    cudaDeviceSynchronize();
    cudaMemcpy(out_data.data(), d_res, mem_bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(fft_plan);
    cufftDestroy(ifft_plan);
    cudaFree(d_in);
    cudaFree(d_buf);
    cudaFree(d_res);

    return out_data;
}