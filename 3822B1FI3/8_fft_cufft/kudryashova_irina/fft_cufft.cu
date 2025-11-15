#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

// scale kernel
__global__ void scale_complex(float2* data, int total_elems, float inv_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) {
        float2 v = data[idx];
        v.x *= inv_n;
        v.y *= inv_n;
        data[idx] = v;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    const size_t floats = input.size();
    const int n = static_cast<int>(floats / (2 * batch));
    const size_t complex_count = static_cast<size_t>(batch) * n;
    const size_t bytes = complex_count * sizeof(cufftComplex);
    std::vector<float> output(floats);
    cudaStream_t compute_stream;
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

    cudaHostRegister(const_cast<float*>(input.data()), bytes, 0);
    cudaHostRegister(output.data(), bytes, 0);

    cufftComplex* dev_data = nullptr;
    cudaMalloc(&dev_data, bytes);

    cudaMemcpyAsync(dev_data, input.data(), bytes, cudaMemcpyHostToDevice, compute_stream);
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    cufftSetStream(plan, compute_stream);

    cufftExecC2C(plan, dev_data, dev_data, CUFFT_FORWARD);
    cufftExecC2C(plan, dev_data, dev_data, CUFFT_INVERSE);

    const float inv_n = 1.0f / static_cast<float>(n);
    int threads = 256;
    int blocks  = static_cast<int>((complex_count + threads - 1) / threads);
    scale_complex<<<blocks, threads, 0, compute_stream>>>(
        reinterpret_cast<float2*>(dev_data),
        static_cast<int>(complex_count),
        inv_n
    );

    cudaMemcpyAsync(output.data(), dev_data, bytes, cudaMemcpyDeviceToHost, compute_stream);
    cudaStreamSynchronize(compute_stream);

    cufftDestroy(plan);
    cudaFree(dev_data);

    cudaHostUnregister(const_cast<float*>(input.data()));
    cudaHostUnregister(output.data());
    cudaStreamDestroy(compute_stream);

    return output;
}