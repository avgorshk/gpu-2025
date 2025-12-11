#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void normalizeKernel(cufftComplex* data, int total, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FftCUFFT(const std::vector<float>& input, int batch) {
    std::vector<float> output(input.size());
    if (input.empty() || batch <= 0) return output;

    int total_complex = input.size() / 2;
    int n = total_complex / batch;
    if (n <= 0) return output;

    cufftComplex* d_data = nullptr;
    size_t bytes = input.size() * sizeof(float);
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    int rank = 1;
    int dims[1] = { n };
    int inembed[1] = { n };
    int onembed[1] = { n };

    cufftPlanMany(&plan, rank, dims,
                  inembed, 1, n,
                  onembed, 1, n,
                  CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int threads = 256;
    int blocks = (total_complex + threads - 1) / threads;
    normalizeKernel<<<blocks, threads>>>(d_data, total_complex, n);

    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}
