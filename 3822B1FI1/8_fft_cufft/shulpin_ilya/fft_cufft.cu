#include "fft_cufft.h"

__global__ void normalize(cufftComplex* data, int total, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    size_t total = input.size();
    size_t N = total / (2 * static_cast<size_t>(batch));

    size_t total_complex = N * static_cast<size_t>(batch);
    int total_batch = static_cast<int>(total_complex);

    size_t bytes = total_complex * sizeof(cufftComplex);
    std::vector<float> out(total);

    float scale = 1.0f / static_cast<float>(N);
    int block = 256;
    int grid = (total_batch + block - 1) / block;

    cufftComplex* data = nullptr;
    cudaMalloc(&data, bytes);
    cudaMemcpy(data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, static_cast<int>(N), CUFFT_C2C, batch);

    cufftExecC2C(plan, data, data, CUFFT_FORWARD);
    cufftExecC2C(plan, data, data, CUFFT_INVERSE);

    normalize<<<grid, block>>>(data, total_batch, scale);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), data, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(data);

    return out;
}