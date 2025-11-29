#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "gemm_cublas.h"

namespace {
template <typename Vec>
void ResizeUninitialized(Vec& v, std::size_t size) {
  struct stub {
    typename Vec::value_type v;
    stub() {}
  };
  reinterpret_cast<std::vector<stub>&>(v).resize(size);
}
}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
  const std::size_t nn = n * n;

  float* devbuf;
  cudaMalloc(&devbuf, 3 * nn * sizeof(*devbuf));
  //
  float* dev_a = devbuf + (0 * nn);
  float* dev_b = devbuf + (1 * nn);
  float* dev_c = devbuf + (2 * nn);

  cudaMemcpy(dev_a, a.data(), nn * sizeof(*dev_a), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b.data(), nn * sizeof(*dev_b), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  //
  const float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle,                    //
              CUBLAS_OP_N, CUBLAS_OP_N,  //
              n, n, n,                   //
              &alpha,                    //
              dev_b, n,                  //
              dev_a, n,                  //
              &beta,                     //
              dev_c, n);
  //

  std::vector<float> c;
  ResizeUninitialized(c, nn);
  cudaMemcpy(c.data(), dev_c, nn * sizeof(*dev_c), cudaMemcpyDeviceToHost);

  //
  cublasDestroy(handle);

  cudaFree(devbuf);

  return c;
}