#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    int total_size = 2 * n * batch;
    
    std::vector<float> output(total_size);
    
    cufftComplex *d_data;
    cudaMalloc(&d_data, total_size * sizeof(float));
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    
    cudaMemcpy(d_data, input.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    float scale = 1.0f / n;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    auto normalize_kernel = [] __global__ (cufftComplex* data, float scale, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx * 2 < size) {
            data[idx].x *= scale;
            data[idx].y *= scale;
        }
    };
    
    normalize_kernel<<<blocks, threads, 0, stream>>>((cufftComplex*)d_data, scale, total_size);
    
    cudaMemcpyAsync(output.data(), d_data, total_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    cufftDestroy(plan);
    cudaFree(d_data);
    
    return output;
}