#include "fft_cufft.h"
#include <cufft.h>

__global__ void normalize_kernel(cufftComplex* data, int n, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    size_t complex_size = n * batch * sizeof(cufftComplex);
    
    std::vector<float> output(input.size());
    cufftComplex* d_data;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMalloc(&d_data, complex_size);
    cudaMemcpyAsync(d_data, input.data(), complex_size, cudaMemcpyHostToDevice, stream);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    cufftSetStream(plan, stream);
    
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    int threads = 256;
    int blocks = (n * batch + threads - 1) / threads;
    normalize_kernel<<<blocks, threads, 0, stream>>>(d_data, n, n * batch);
    
    cudaMemcpyAsync(output.data(), d_data, complex_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    cufftDestroy(plan);
    cudaFree(d_data);
    cudaStreamDestroy(stream);
    
    return output;
}