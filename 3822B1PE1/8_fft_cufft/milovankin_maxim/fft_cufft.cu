#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

__global__ void scale_complex(float* data, int n, int batch_sz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float scale_factor = 1.0f / static_cast<float>(n);
        data[idx].x *= scale_factor;
        data[idx].y *= scale_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    size_t input_sz = input.size();
    
    if (input_sz == 0 || batch <= 0) {
        return {};
    }
    
    if (input_sz % (2 * batch) != 0) {
        return {};
    }

    int fft_len = static_cast<int>(input_sz / (2 * batch));
    size_t complex_cnt = static_cast<size_t>(fft_len) * batch;
    size_t dev_mem = complex_cnt * sizeof(cufftComplex);

    cufftComplex* gpu_data = nullptr;
    if (cudaMalloc(&gpu_data, dev_mem) != cudaSuccess) {
        return {};
    }

    if (cudaMemcpy(gpu_data, input.data(), dev_mem, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(gpu_data);
        return {};
    }

    cufftHandle fft_plan;
    if (cufftPlan1d(&fft_plan, fft_len, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        cudaFree(gpu_data);
        return {};
    }

    // Execute forward transform
    if (cufftExecC2C(fft_plan, gpu_data, gpu_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        cufftDestroy(fft_plan);
        cudaFree(gpu_data);
        return {};
    }

    // Execute inverse transform
    if (cufftExecC2C(fft_plan, gpu_data, gpu_data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        cufftDestroy(fft_plan);
        cudaFree(gpu_data);
        return {};
    }

    // Normalize the result on the GPU
    int threads = 256;
    int blocks = (static_cast<int>(complex_cnt) + threads - 1) / threads;
    scale_complex<<<blocks, threads>>>(gpu_data, fft_len, static_cast<int>(complex_cnt));

    cudaDeviceSynchronize();

    std::vector<float> output(input_sz);
    if (cudaMemcpy(output.data(), gpu_data, dev_mem, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cufftDestroy(fft_plan);
        cudaFree(gpu_data);
        return {};
    }

    cufftDestroy(fft_plan);
    cudaFree(gpu_data);

    return output;
}
