#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {


    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> host_c(n * n);

    cudaStream_t compute_stream;
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

    cudaHostRegister(const_cast<float*>(a.data()), bytes, 0);
    cudaHostRegister(const_cast<float*>(b.data()), bytes, 0);
    cudaHostRegister(host_c.data(), bytes, 0);

    float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    cudaMalloc(&dev_a, bytes);
    cudaMalloc(&dev_b, bytes);
    cudaMalloc(&dev_c, bytes);

    cudaMemcpyAsync(dev_a, a.data(), bytes, cudaMemcpyHostToDevice, compute_stream);
    cudaMemcpyAsync(dev_b, b.data(), bytes, cudaMemcpyHostToDevice, compute_stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, compute_stream);
    (void)cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    const float one  = 1.0f;
    const float zero = 0.0f;
    int side = n;

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n,  // m
        n,  // n
        n,  // k
        &one,
        dev_b, side, 
        dev_a, side,
        &zero,
        dev_c, side
    );

    cudaMemcpyAsync(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost, compute_stream);
    cudaStreamSynchronize(compute_stream);

    cublasDestroy(handle);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaHostUnregister(const_cast<float*>(a.data()));
    cudaHostUnregister(const_cast<float*>(b.data()));
    cudaHostUnregister(host_c.data());
    cudaStreamDestroy(compute_stream);

    return host_c;
}