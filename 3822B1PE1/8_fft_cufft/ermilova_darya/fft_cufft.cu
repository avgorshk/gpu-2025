#include "fft_cufft.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdexcept>
#include <string>

inline void cudaCheck(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

inline void cufftCheck(cufftResult res, const char* msg)
{
    if (res != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("cuFFT: ") + msg);
    }
}

__global__ void normalizeKernel(cufftComplex* data,
    int totalComplex,
    float n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalComplex) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch)
{
    if (batch <= 0) {
        return {};
    }

    int n = 0;

    if (input.size() % (2 * batch) != 0) {
        throw std::runtime_error("FffCUFFT: wrong input size for given batch");
    }
    n = static_cast<int>(input.size() / (2 * batch));

    const int totalComplex = n * batch;
    const size_t bytes = sizeof(float) * input.size();

    cufftComplex* d_data = nullptr;
    cudaCheck(cudaMalloc(&d_data, bytes), "cudaMalloc d_data failed");

    cudaCheck(cudaMemcpy(d_data, input.data(), bytes,
        cudaMemcpyHostToDevice),
        "Memcpy input H2D failed");

    cufftHandle plan;
    int rank = 1;
    int nArr[1] = { n };
    int istride = 1, ostride = 1;
    int idist = n, odist = n;
    int inembed[1] = { n };
    int onembed[1] = { n };

    cufftCheck(
        cufftPlanMany(&plan,
            rank,
            nArr,
            inembed, istride, idist,
            onembed, ostride, odist,
            CUFFT_C2C,
            batch),
        "cufftPlanMany failed");

    cufftCheck(
        cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD),
        "cufftExecC2C forward failed");

    cufftCheck(
        cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE),
        "cufftExecC2C inverse failed");

    {
        int blockSize = 256;
        int gridSize = (totalComplex + blockSize - 1) / blockSize;
        normalizeKernel << <gridSize, blockSize >> > (d_data, totalComplex, static_cast<float>(n));
        cudaCheck(cudaGetLastError(), "normalizeKernel launch failed");
    }

    std::vector<float> output(input.size());
    cudaCheck(cudaMemcpy(output.data(), d_data, bytes,
        cudaMemcpyDeviceToHost),
        "Memcpy output D2H failed");

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}
