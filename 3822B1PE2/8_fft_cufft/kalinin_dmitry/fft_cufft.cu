#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>

static cufftHandle plan_forward = 0;
static cufftHandle plan_inverse = 0;
static int cached_n = 0;
static int cached_batch = 0;
static bool plans_initialized = false;

__global__ void normalize_kernel(cufftComplex* data, float norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= norm;
        data[idx].y *= norm;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    
    size_t complex_size = n * batch * sizeof(cufftComplex);
    std::vector<float> output(input.size());
    
    if (!plans_initialized || cached_n != n || cached_batch != batch) {
        if (plans_initialized) {
            cufftDestroy(plan_forward);
            cufftDestroy(plan_inverse);
        }
        cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch);
        cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch);
        cached_n = n;
        cached_batch = batch;
        plans_initialized = true;
    }
    
    cufftComplex *d_data;
    cudaMalloc(&d_data, complex_size);
    
    cudaMemcpyAsync(d_data, input.data(), complex_size, cudaMemcpyHostToDevice, 0);
    
    cufftExecC2C(plan_forward, d_data, d_data, CUFFT_FORWARD);
    
    cufftExecC2C(plan_inverse, d_data, d_data, CUFFT_INVERSE);
    
    float norm = 1.0f / static_cast<float>(n);
    int total_elements = n * batch;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    normalize_kernel<<<numBlocks, blockSize>>>(d_data, norm, total_elements);
    
    cudaDeviceSynchronize();
    
    cudaMemcpyAsync(output.data(), d_data, complex_size, cudaMemcpyDeviceToHost, 0);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    
    return output;
}

