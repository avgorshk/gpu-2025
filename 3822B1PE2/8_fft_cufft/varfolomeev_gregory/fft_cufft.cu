#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void normalize_kernel(cufftComplex* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.empty() || batch <= 0) {
        return {};
    }
    
    int n = input.size() / (2 * batch);
    if (n <= 0) {
        return {};
    }
    
    int total_complex = n * batch;
    size_t bytes = total_complex * sizeof(cufftComplex);
    
    std::vector<float> output(input.size());
    
    cufftComplex* d_data;
    cudaMalloc(&d_data, bytes);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cufftSetStream(plan, stream);
    
    // Async memory copy for overlap
    cudaMemcpyAsync(d_data, input.data(), input.size() * sizeof(float), 
                    cudaMemcpyHostToDevice, stream);
    
    // Forward FFT (in-place)
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    
    // Inverse FFT (in-place)
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    // Normalize on device
    float scale = 1.0f / n;
    int block_size = 256;
    int grid_size = (total_complex + block_size - 1) / block_size;
    normalize_kernel<<<grid_size, block_size, 0, stream>>>(d_data, total_complex, scale);
    
    // Async memory copy for overlap
    cudaMemcpyAsync(output.data(), d_data, input.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cufftDestroy(plan);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    
    return output;
}

