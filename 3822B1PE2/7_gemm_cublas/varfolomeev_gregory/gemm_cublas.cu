#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }
    
    size_t matrix_size = static_cast<size_t>(n) * n;
    size_t bytes = matrix_size * sizeof(float);
    
    if (a.size() != matrix_size || b.size() != matrix_size) {
        return {};
    }
    
    std::vector<float> c(matrix_size);
    
    float *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    
    // Async memory copies for overlap
    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS uses column-major, input is row-major
    // C = A * B (row-major) is equivalent to C^T = B^T * A^T (column-major)
    // So we use CUBLAS_OP_T for both matrices and swap order: d_b, d_a
    cublasSgemm(handle,
                CUBLAS_OP_T,  // Transpose B (B^T in column-major = B in row-major)
                CUBLAS_OP_T,  // Transpose A (A^T in column-major = A in row-major)
                n, n, n,
                &alpha,
                d_b, n,  // B^T
                d_a, n,  // A^T
                &beta,
                d_c, n); // C^T (result in column-major)
    
    // Result is in column-major, need to transpose back to row-major
    std::vector<float> c_colmajor(matrix_size);
    cudaMemcpyAsync(c_colmajor.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // Transpose result from column-major to row-major
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i * n + j] = c_colmajor[j * n + i];
        }
    }
    
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}


