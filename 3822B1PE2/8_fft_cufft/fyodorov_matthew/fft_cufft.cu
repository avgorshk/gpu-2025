#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch)
{
    if (input.empty() || batch <= 0)
    {
        throw std::invalid_argument("Неверные параметры входных данных");
    }

    int total_elements = input.size();
    if (total_elements % (2 * batch) != 0)
    {
        throw std::invalid_argument("Неверный размер входного массива");
    }

    int n = total_elements / (2 * batch);
    std::vector<float> output(total_elements, 0.0f);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cufftComplex *d_input = nullptr, *d_fft = nullptr;
    size_t data_size = total_elements * sizeof(float);

    cudaError_t cuda_err;
    cuda_err = cudaMallocAsync(&d_input, data_size, stream);
    if (cuda_err != cudaSuccess)
    {
        cudaStreamDestroy(stream);
        throw std::runtime_error("Ошибка выделения памяти для входных данных: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cuda_err = cudaMallocAsync(&d_fft, data_size, stream);
    if (cuda_err != cudaSuccess)
    {
        cudaFreeAsync(d_input, stream);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Ошибка выделения памяти для FFT данных: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cufftComplex *h_input_complex = reinterpret_cast<cufftComplex *>(const_cast<float *>(input.data()));
    cuda_err = cudaMemcpyAsync(d_input, h_input_complex, data_size,
                               cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess)
    {
        cleanup_fft(d_input, d_fft, stream, nullptr);
        throw std::runtime_error("Ошибка копирования входных данных: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cufftHandle plan_forward, plan_inverse;
    cufftResult_t cufft_err;

    cufft_err = cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch);
    if (cufft_err != CUFFT_SUCCESS)
    {
        cleanup_fft(d_input, d_fft, stream, nullptr);
        throw std::runtime_error("Ошибка создания плана прямого FFT");
    }

    cufft_err = cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch);
    if (cufft_err != CUFFT_SUCCESS)
    {
        cufftDestroy(plan_forward);
        cleanup_fft(d_input, d_fft, stream, nullptr);
        throw std::runtime_error("Ошибка создания плана обратного FFT");
    }

    cufftSetStream(plan_forward, stream);
    cufftSetStream(plan_inverse, stream);

    cufft_err = cufftExecC2C(plan_forward, d_input, d_fft, CUFFT_FORWARD);
    if (cufft_err != CUFFT_SUCCESS)
    {
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        cleanup_fft(d_input, d_fft, stream, nullptr);
        throw std::runtime_error("Ошибка выполнения прямого FFT");
    }

    cufft_err = cufftExecC2C(plan_inverse, d_fft, d_input, CUFFT_INVERSE);
    if (cufft_err != CUFFT_SUCCESS)
    {
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        cleanup_fft(d_input, d_fft, stream, nullptr);
        throw std::runtime_error("Ошибка выполнения обратного FFT");
    }

    int total_threads = 256;
    int total_blocks = (total_elements / 2 + total_threads - 1) / total_threads;

    normalize_kernel<<<total_blocks, total_threads, 0, stream>>>(
        reinterpret_cast<float2 *>(d_input), n, batch);

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess)
    {
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        cleanup_fft(d_input, d_fft, stream, nullptr);
        throw std::runtime_error("Ошибка выполнения ядра нормализации: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cufftComplex *h_output_complex = reinterpret_cast<cufftComplex *>(output.data());
    cuda_err = cudaMemcpyAsync(h_output_complex, d_input, data_size,
                               cudaMemcpyDeviceToHost, stream);
    if (cuda_err != cudaSuccess)
    {
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        cleanup_fft(d_input, d_fft, stream, nullptr);
        throw std::runtime_error("Ошибка копирования результата: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cudaStreamSynchronize(stream);

    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cleanup_fft(d_input, d_fft, stream, nullptr);

    return output;
}

__global__ void normalize_kernel(float2 *data, int n, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_complex = n * batch;

    if (idx < total_complex)
    {
        float scale = 1.0f / n;
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

void cleanup_fft(cufftComplex *d_input, cufftComplex *d_fft,
                 cudaStream_t stream, cufftHandle *plans, int num_plans = 0)
{
    if (d_input)
        cudaFreeAsync(d_input, stream);
    if (d_fft)
        cudaFreeAsync(d_fft, stream);
    if (stream)
        cudaStreamDestroy(stream);
}