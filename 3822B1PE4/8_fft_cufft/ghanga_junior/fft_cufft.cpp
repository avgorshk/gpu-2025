#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> FftCuFFT(const std::vector<float>& input, int n)
{
    // Sortie R2C : n/2 + 1 valeurs complexes => 2*(n/2+1) floats
    int out_size = (n / 2 + 1);
    std::vector<float> output(out_size * 2, 0.0f);

    if (n <= 0 || input.size() != static_cast<size_t>(n)) {
        return output;
    }

    // Pointeurs GPU
    float* d_input  = nullptr;
    cufftComplex* d_output = nullptr;

    cudaMalloc(&d_input,  sizeof(float) * n);
    cudaMalloc(&d_output, sizeof(cufftComplex) * out_size);

    cudaMemcpy(d_input, input.data(),
               sizeof(float) * n,
               cudaMemcpyHostToDevice);

    // Création plan cuFFT
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_R2C, 1);

    // Exécution FFT
    cufftExecR2C(plan, d_input, d_output);

    cudaDeviceSynchronize();

    // Copie device -> host (complexes interleavés)
    cudaMemcpy(output.data(), d_output,
               sizeof(cufftComplex) * out_size,
               cudaMemcpyDeviceToHost);

    // Libération
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
