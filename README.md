# Content
- [How To](#how-to)
- [Configuration](#configuration)
- [Time Measurement](#time-measurement)
- [Tasks](#tasks)
- [Results](#results)

# How To
1. Create [github](https://github.com/) account (if not exists);
2. Make sure SSH clone & commit is working ([Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh));
3. Fork this repo (just click **Fork** button on the top of the page, detailed instructions [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project))
4. Clone your forked repo into your local machine, use your user instead of `username`:
```sh
git clone git@github.com:username/gpu-2025.git
cd gpu-2025
```
5. Go to your group folder, e.g.:
```sh
cd 3822B1FI1
```
6. Go to needed task folder, e.g.:
```sh
cd 1_gelu_omp
```
7. Create new folder with your surname and name (**make sure it's the same for all tasks**), e.g.:
```sh
mkdir petrov_ivan
```
8. Copy your task source/header files (including main program) into this folder (use `copy` instead of `cp` on Windows), e.g.:
```sh
cd petrov_ivan
cp /home/usr/lab/*.cpp .
cp /home/usr/lab/*.h .
```
8. Push your sources to github repo, e.g.:
```sh
cd ..
git add .
git commit -m "1_gelu_omp task"
git push
```
9. Go to your repo in browser, click **Contribute** button on the top of page, then **Open pull request**. Provide meaningfull request title and description, then **Create pull request** (see details [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)).
10. Go to Pull Requests [page](https://github.com/avgorshk/gpu-2025/pulls) in course repo, find your pull request and check if there are no any merge conflicts occur. If merge conflicts happen - resolve it following the instruction provided by github.

# Time Measurement
The following scheme is used to measure task execution time:
```cpp
int main() {
    // ...

    // Warming-up
    Task(input, size);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        Task(input, size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    // ...
}
```

# Configuration
- CPU: Intel Core i5 12600K (4 cores, 4 threads)
- RAM: 16 GB
- GPU: NVIDIA RTX 4060 (8 GB)
- OS:  Ubuntu 22.04.3 LTS
- Host Compiler: GCC 11.4.0 (C++17)
- CUDA: 12.9

# Tasks
## Task #1: OpenMP GELU Implementation
The **Gaussian Error Linear Unit (GELU)** is an activation function frequently used in Deep Neural Networks (DNNs) and can be thought of as a smoother ReLU.

To approximate GELU function, use the following formula:

GELU(x) =  $0.5x(1 + tanh(\sqrt{2 / \pi}(x + 0.044715 * x^3)))$

Implement the function with the following interface in C++:
```cpp
std::vector<float> GeluOMP(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use OpenMP technology to make your function parallel & fast.

Two files are expected to be uploaded:
- gelu_omp.h
```cpp
#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input);

#endif // __GELU_OMP_H
```
- gelu_omp.cpp
```cpp
#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    // Place your implementation here
}
```
**Performance Hints:**
 - better formula to compute GELU, e.g. replace *tanh()* with *exp()*;
 - loop unrolling;
 - loop vectorization;
 - vector allocation and computations in different threads *(Windows only)*.

## Task #2: CUDA GELU Implementation
Implement the function with the following interface in CUDA C++ using the formula described above:
```cpp
std::vector<float> GeluCUDA(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use CUDA technology to make your function work on NVIDIA GPU. Try to make it fast.

Two files are expected to be uploaded:
- gelu_cuda.h
```cpp
#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#include <vector>

std::vector<float> GeluCUDA(const std::vector<float>& input);

#endif // __GELU_CUDA_H
```
- gelu_cuda.cu
```cpp
#include "gelu_cuda.h"

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    // Place your implementation here
}
```
**Performance Hints:**
 - overlap host memory allocation and CUDA computations;
 - allocate and free device memory once;
 - use better formula to compute GELU, e.g. replace *tanh()* with *exp()*.

## Task #3: Naive Matrix Multiplication using OpenMP
General matrix multiplication (GEMM) is a very basic and broadly used linear algebra operation applied in high performance computing (HPC), statistics, deep learning and other domains. There are a lot of GEMM algorithms with different mathematical complexity form $O(n^3)$ for naive and block approaches to $O(n^{2.371552})$ for the method descibed by Williams et al. in 2024 [[1](https://epubs.siam.org/doi/10.1137/1.9781611977912.134)]. But despite a variety of algorithms with low complexity, block matrix multiplication remains the most used implementation in practice since it fits to modern HW better.

To start learning matrix multiplication smoother, let us start with naive approach here. To compute matrix multiplication result C for matricies A and B, where C = A * B and the size for all matricies are $n*n$, one should use the following formula for each element of C (will consider only square matricies for simplicity):

$c_{ij}=\sum_{k=1}^na_{ik}b_{kj}$

To complete the task one should implement a function that multiplies two square matricies using OpenMP with the following interface:
```cpp
std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n);
```
Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- naive_gemm_omp.h:
```cpp
#ifndef __NAIVE_GEMM_OMP_H
#define __NAIVE_GEMM_OMP_H

#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n);

#endif // __NAIVE_GEMM_OMP_H
```
- naive_gemm_omp.cpp:
```cpp
#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    // Place your implementation here
}
```
**Performance Hints:**
 - cache-friendly memory accesses;
 - loop unrolling;
 - loop vectorization.

## Task #4: Naive Matrix Multiplication using CUDA
In this task one should implement naive approach for matrix multiplication in CUDA trying to make it fast enough *(pay attention to global memory accesses in your code)*.

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- naive_gemm_cuda.h:
```cpp
#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __NAIVE_GEMM_CUDA_H
```
- naive_gemm_cuda.cu:
```cpp
#include "naive_gemm_cuda.h"

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    // Place your implementation here
}
```
**Performance Hints:**
 - warp-friendly memory accesses;
 - multiple elements per warp processing;
 - loop unrolling and memory load vectorization;
 - block size selection;
 - overlap host memory allocation and CUDA computations.

## Task #5: Block Matrix Multiplication using OpenMP
In real applications block-based approach for matrix multiplication can get multiple times faster execution comparing with naive version due to cache friendly approach. To prove this in practice, implement such a version in C++ using OpenMP.

In block version algorithm could be divided into three stages:
1. Split matricies into blocks (block size normally affects performance significantly so choose it consciously);
2. Multiply two blocks to get partial result;
3. Replay step 2 for all row/column blocks accumulating values into a single result block.

From math perspective, block matrix multiplication could be described by the following formula, where $C_{IJ}$, $A_{IK}$ and $B_{KJ}$ are sub-matricies with the size $block\_size*block\_size$:

$C_{IJ}=\sum_{k=1}^{block_count}A_{IK}B_{KJ}$

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- block_gemm_omp.h:
```cpp
#ifndef __BLOCK_GEMM_OMP_H
#define __BLOCK_GEMM_OMP_H

#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n);

#endif // __BLOCK_GEMM_OMP_H
```
- block_gemm_omp.cpp:
```cpp
#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    // Place your implementation here
}
```

As in previous task, let us consider all matricies are square.

**Performance Hints:**
 - cache-friendly memory accesses;
 - loop unrolling;
 - loop vectorization.

## Task #6: Block Matrix Multiplication using CUDA
In CUDA C++ block-based approach looks similar. But to get better performance one should use CUDA shared memory to store each particular block while computations. With this consideration, algorithm will be the following:
1. A single CUDA block should compute a single block of result matrix C, a single CUDA thread - a single matrix C element;
2. For each A block in a row and B block in a column:
    1. Load A block into shared memory;
    2. Load B block into shared memory;
    3. Synchronize over all threads in block;
    4. Compute BlockA * BlockB and accumulate into C block in shared memory;
    5. Synchronize over all threads in block;
3. Dump block C from shared to global memory.

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- block_gemm_cuda.h:
```cpp
#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __BLOCK_GEMM_CUDA_H
```
- block_gemm_cuda.cu:
```cpp
#include "block_gemm_cuda.h"

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    // Place your implementation here
}
```
**Performance Hints:**
 - shared memory usage to store matrix block;
 - warp-friendly memory accesses;
 - multiple elements per warp processing;
 - loop unrolling and memory load vectorization;
 - block size selection;
 - overlap host memory allocation and CUDA computations.

## Task #7: Matrix Multiplication using cuBLAS
The most performant way to multiply two matrices on particular hardware is to use vendor-provided library for this purpose. In CUDA it's [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html). Try to use cuBLAS API to implement general matrix multiplication in most performant way.

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Note, that in cuBLAS API matrix is expected to be stored by columns, so additional transpose may be required.

Two files are expected to be uploaded:
- gemm_cublas.h:
```cpp
#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n);

#endif // __GEMM_CUBLAS_H
```
- gemm_cublas.cu:
```cpp
#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    // Place your implementation here
}
```
**Performance Hints:**
 - overlap host memory allocation and CUDA computations;
 - avoid redundant device memory allocation.

## Task #8: FFT (Fast Fourier Transform) using cuFFT
Another widely used operation in HPC & signal processing is discrete [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform). Naive approach (by definition) has $O(n^2)$ complexity and is not used in practice due to its slowness. Better way is [Fast Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform) algorithm with $O(n*log(n))$ complexity.

Due to its frequent use, FFT algorithm implementation is normally a part of vendor-optimized solutions for various hardware chips. For NVIDIA GPUs one should take [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) library.

To pass the task one should implement a funtion that takes $batch$ signals of $n$ complex elements, and performs complex-to-complex forward and than inverse Fourier transform for them. For better performance use cuFFT API.

Required function should have the following prototype:
```cpp
std::vector<float> FffCUFFT(const std::vector<float>& input, int batch);
```
Here $batch$ is a number of independent signals, $input$ contains complex values in the format of $(real, imaginary)$ pairs of floats storing pair by pair. So $input$ array size must be equal to $2 * n * batch$.

The function should perform the following actions:
1. Compute forward Fourier transform for $input$;
2. Compute inverse Fourier transform for the result of step 1;
3. Normalize result of step 2 by $n$.

Returned array must store result of step 3 in the same format of $(real, imaginary)$ pairs as $input$ and have the same size.

Note, that due to Fourier Transform math properties, result array will have the same values as input one. This specificity could be used for self-checking.

Two files are expected to be uploaded:
- fft_cufft.h:
```cpp
#ifndef __FFT_CUFFT_H
#define __FFT_CUFFT_H

#include <vector>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch);

#endif // __FFT_CUFFT_H
```
- fft_cufft.cu:
```cpp
#include "fft_cufft.h"

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    // Place your implementation here
}
```
**Performance Hints:**
 - make normalization on device;
 - do not allocate redundant device memory;
 - overlap host memory allocation and CUDA computations.

## Task #9: OpenCL GELU Implementation
Implement GELU function with the following interface in OpenCL using the formula described in task #1:
```cpp
std::vector<float> GeluOCL(const std::vector<float>& input, int platform);
```
Size of result vector should be the same as for `input`. Use OpenCL technology to make your function work on NVIDIA GPU. Try to make it fast.

Use `CL_DEVICE_GPU` flag to choose GPU device. Use `platform` platform and `0` device. Store your OpenCL kernel in a string constant.

Two files are expected to be uploaded:
- gelu_ocl.h
```cpp
#ifndef __GELU_OCL_H
#define __GELU_OCL_H

#include <vector>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform);

#endif // __GELU_OCL_H
```
- gelu_ocl.cpp
```cpp
#include "gelu_ocl.h"

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    // Place your implementation here
}
```
**Performance Hints:**
 - perform OpenCL boilerplate code once;
 - use better formula to compute GELU, e.g. replace *tanh()* with *exp()*;
 - overlap host memory allocation and GPU computations.

# Results
## 1_gelu_omp (134217728 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|**FAST**|**FAST**|**0.1136**|**-**|
|3822B1PE1|krylov_mikhail|0.1947|12|
|3822B1FI1|ionova_ekaterina|0.2465|6|
|3822B1FI1|Ionova_ekaterina|0.2479|4|
|3822B1PE3|oturin_alexander|0.2526|2|
|3822B1PE1|rams_sergei|0.2559|5|
|3822B1FI1|elvin_veliev|0.2598|3|
|3822B1FI1|shulpin_ilya|0.2636|2|
|3822B1FI3|kholin_kirill|0.2718|2|
|3822B1FI3|kirill_kholin|0.2733|4|
|3822B1FI1|komshina_daria|0.2744|5|
|3822B1PE1|tyurin_mikhail|0.2772|8|
|3822B1PE2|muradov_mike|0.2782|7|
|3822B1PE4|zinoviev_alexander|0.2783|4|
|3822B1PE2|ermolaev_vladislav|0.2792|3|
|3822B1PE2|filatieva_elizaveta|0.2795|8|
|3822B1FI3|kudryashova_irina|0.2810|1|
|3822B1PE1|kalyakina_anastasia|0.2811|15|
|3822B1PE3|sotskov_andrey|0.2822|1|
|3822B1PE2|mukhina_margarita|0.2859|5|
|3822B1PE2|korovin_nikita|0.2914|2|
|3822B1PE1|khasanyanov_kirill|0.2921|17|
|3822B1FI3|koshkin_nikita|0.2933|5|
|3822B1PE1|vershinina_alexandra|0.2956|16|
|3822B1PE4|podovinnikov_artyom|0.3142|3|
|3822B1PE2|filatiev_vladislav|0.4155|9|
|**REF**|**REF**|**0.4736**|**-**|
|3822B1FI2|yasakova_tanya|0.6437|1|
|3822B1PE4|karaseva_ekaterina|0.6467|5|
|3822B1PE1|polikanov_vitaliy|0.6471|18|
|3822B1PE1|sidorina_polina|0.6509|19|
|3822B1FI3|lavrentyev_alexey|0.6559|6|
|3822B1PE3|kazunin_nikita|0.6626|4|
|3822B1PE2|titov_semyon|0.6721|1|
|3822B1PE1|konstantinov_ilya|0.6725|9|
|3822B1PE1|moiseev_artem|0.6731|3|
|3822B1PE4|shuravina_oksana|0.6742|2|
|3822B1PE1|korablev_vladlen|0.6773|1|
|3822B1PE4|kolokolova_darya|0.6778|1|
|3822B1PE1|morozov_egor|0.6782|4|
|3822B1FI1|solovev_alexey|0.6786|1|
|3822B1PE1|gnitienko_kirill|0.6802|10|
|3822B1PE1|milovankin_maxim|0.6830|11|
|3822B1PE2|sorokin_andrey|0.6830|6|
|3822B1PE1|shvedova_vitalina|0.6866|2|
|3822B1PE2|kondratev_yaroslav|0.6881|4|
|3822B1PE1|ermilova_darya|0.6957|14|
|3822B1PE1|nikolaev_roman|0.6993|13|
|3822B1FI3|solovyev_danila|0.7275|3|
|3822B1PE1|sadikov_ivan|0.7339|6|
|3822B1PE3|sarafanov_maxim|0.7546|3|
|3822B1PE1|odintsov_misha|0.7605|7|
|3822B1PE2|vladimirova_julia|TEST FAILED|-|
|3822B1FI2|sdobnov_vladimir|BUILD FAILED|-|

## 2_gelu_cuda (134217728 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|**FAST**|**FAST**|**0.1205**|**-**|
|3822B1PE4|shuravina_oksana|0.1840|1|
|3822B1FI1|solovev_alexey|0.2105|1|
|3822B1PE1|polikanov_vitaliy|0.2118|18|
|3822B1FI2|yasakova_tanya|0.2151|1|
|3822B1FI3|lavrentyev_alexey|0.2153|4|
|3822B1PE1|shvedova_vitalina|0.2161|5|
|3822B1PE1|tyurin_mikhail|0.2190|8|
|3822B1FI1|ionova_ekaterina|0.2191|5|
|3822B1PE1|ermilova_darya|0.2195|13|
|3822B1PE1|khasanyanov_kirill|0.2199|17|
|3822B1PE2|ermolaev_vladislav|0.2200|1|
|3822B1PE3|kazunin_nikita|0.2204|3|
|3822B1PE2|filatieva_elizaveta|0.2206|8|
|**REF**|**REF**|**0.2209**|**-**|
|3822B1PE1|milovankin_maxim|0.2217|10|
|3822B1FI1|elvin_veliev|0.2220|3|
|3822B1FI1|shulpin_ilya|0.2222|2|
|3822B1PE2|filatiev_vladislav|0.2224|9|
|3822B1PE4|podovinnikov_artyom|0.2227|3|
|3822B1PE1|gnitienko_kirill|0.2228|9|
|3822B1PE4|zinoviev_alexander|0.2233|4|
|3822B1PE2|kondratev_yaroslav|0.2234|4|
|3822B1PE2|titov_semyon|0.2237|2|
|3822B1PE3|oturin_alexander|0.2239|2|
|3822B1PE4|karaseva_ekaterina|0.2244|5|
|3822B1FI3|solovyev_danila|0.2248|3|
|3822B1PE2|mukhina_margarita|0.2250|5|
|3822B1FI3|koshkin_nikita|0.2257|5|
|3822B1PE1|rams_sergei|0.2259|3|
|3822B1PE2|korovin_nikita|0.2259|3|
|3822B1PE1|sadikov_ivan|0.2262|6|
|3822B1PE3|sotskov_andrey|0.2269|1|
|3822B1FI3|kudryashova_irina|0.2274|1|
|3822B1PE4|kolokolova_darya|0.2279|2|
|3822B1PE1|korablev_vladlen|0.2295|2|
|3822B1PE1|moiseev_artem|0.2298|1|
|3822B1PE2|muradov_mike|0.2306|7|
|3822B1PE2|sorokin_andrey|0.2306|6|
|3822B1PE1|konstantinov_ilya|0.2308|7|
|3822B1PE1|kalyakina_anastasia|0.2311|15|
|3822B1PE1|odintsov_misha|0.2321|14|
|3822B1FI3|kirill_kholin|0.2325|2|
|3822B1PE1|morozov_egor|0.2372|4|
|3822B1PE1|nikolaev_roman|0.2395|12|
|3822B1PE1|krylov_mikhail|0.2577|11|
|3822B1PE1|vershinina_alexandra|0.6209|16|
|3822B1FI1|komshina_daria|0.6698|4|
|3822B1PE2|vladimirova_julia|BUILD FAILED|-|
|3822B1PE3|sarafanov_maxim|TEST FAILED|-|

## 3_naive_gemm_omp (1024 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1PE4|zinoviev_alexander|0.0254|4|
|3822B1PE1|korablev_vladlen|0.0260|4|
|3822B1PE1|milovankin_maxim|0.0261|11|
|3822B1PE1|moiseev_artem|0.0261|3|
|**FAST**|**FAST**|**0.0261**|**-**|
|3822B1PE3|oturin_alexander|0.0264|2|
|3822B1PE2|muradov_mike|0.0265|7|
|3822B1FI3|kirill_kholin|0.0266|2|
|3822B1PE1|tyurin_mikhail|0.0308|8|
|3822B1PE1|ermilova_darya|0.0312|14|
|3822B1PE1|konstantinov_ilya|0.0325|7|
|3822B1PE1|nikolaev_roman|0.0907|12|
|3822B1PE3|sotskov_andrey|0.0940|1|
|3822B1PE2|sorokin_andrey|0.1073|6|
|3822B1PE2|titov_semyon|0.1089|1|
|3822B1PE1|krylov_mikhail|0.1134|9|
|3822B1PE1|rams_sergei|0.1143|5|
|3822B1PE1|kalyakina_anastasia|0.1230|16|
|3822B1PE1|sadikov_ivan|0.1249|1|
|3822B1PE4|karaseva_ekaterina|0.1440|5|
|3822B1PE3|kazunin_nikita|0.1819|3|
|3822B1PE1|morozov_egor|0.1872|6|
|3822B1PE1|vershinina_alexandra|0.1913|17|
|3822B1PE3|sarafanov_maxim|0.2487|4|
|3822B1FI3|lavrentyev_alexey|0.2532|3|
|3822B1FI1|komshina_daria|0.2545|4|
|3822B1PE2|mukhina_margarita|0.2602|4|
|3822B1PE2|korovin_nikita|0.3160|5|
|3822B1FI3|kudryashova_irina|0.6950|1|
|3822B1FI1|ionova_ekaterina|0.7003|5|
|3822B1PE1|khasanyanov_kirill|0.7031|15|
|3822B1FI1|elvin_veliev|0.7035|3|
|3822B1FI3|koshkin_nikita|0.7102|4|
|3822B1PE4|podovinnikov_artyom|0.7137|1|
|3822B1PE2|filatiev_vladislav|0.7226|9|
|3822B1FI2|yasakova_tanya|0.7227|1|
|3822B1PE4|shuravina_oksana|0.7258|2|
|3822B1PE4|kolokolova_darya|0.7264|3|
|3822B1PE2|ermolaev_vladislav|0.7310|2|
|3822B1PE1|gnitienko_kirill|0.7319|10|
|3822B1PE2|filatieva_elizaveta|0.7385|8|
|3822B1FI1|solovev_alexey|0.7486|1|
|3822B1PE2|kondratev_yaroslav|0.7529|3|
|**REF**|**REF**|**0.7773**|**-**|
|3822B1PE1|shvedova_vitalina|0.7832|2|
|3822B1FI1|shulpin_ilya|0.7960|2|
|3822B1PE1|odintsov_misha|0.8459|13|

## 4_naive_gemm_cuda (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1PE1|nikolaev_roman|0.0626|10|
|3822B1PE3|oturin_alexander|0.0638|3|
|**FAST**|**FAST**|**0.0776**|**-**|
|3822B1PE1|krylov_mikhail|0.1392|7|
|3822B1FI2|yasakova_tanya|0.1406|1|
|3822B1FI1|shulpin_ilya|0.1419|2|
|3822B1PE2|filatiev_vladislav|0.1506|9|
|3822B1PE2|korovin_nikita|0.1543|5|
|3822B1FI3|koshkin_nikita|0.1565|3|
|3822B1PE3|sotskov_andrey|0.1604|1|
|3822B1PE1|konstantinov_ilya|0.1638|6|
|3822B1PE4|podovinnikov_artyom|0.1717|1|
|3822B1PE1|ermilova_darya|0.1723|15|
|3822B1PE1|shvedova_vitalina|0.1729|4|
|3822B1PE2|mukhina_margarita|0.1731|4|
|3822B1PE4|kolokolova_darya|0.1755|4|
|3822B1PE1|sadikov_ivan|0.1762|5|
|3822B1PE2|kondratev_yaroslav|0.1763|3|
|3822B1FI3|kirill_kholin|0.1816|1|
|3822B1FI3|kudryashova_irina|0.1833|2|
|3822B1PE2|muradov_mike|0.1834|7|
|3822B1PE1|tyurin_mikhail|0.1837|11|
|3822B1PE2|ermolaev_vladislav|0.1841|2|
|3822B1PE1|kalyakina_anastasia|0.1853|14|
|3822B1PE1|vershinina_alexandra|0.1856|16|
|3822B1PE4|shuravina_oksana|0.1955|2|
|3822B1FI1|komshina_daria|0.2030|4|
|3822B1PE3|kazunin_nikita|0.2165|2|
|3822B1PE1|milovankin_maxim|0.2353|8|
|3822B1PE4|karaseva_ekaterina|0.2430|5|
|3822B1FI3|lavrentyev_alexey|0.2466|4|
|3822B1FI1|elvin_veliev|0.2481|3|
|3822B1PE2|filatieva_elizaveta|0.2492|8|
|3822B1PE1|moiseev_artem|0.2539|2|
|3822B1PE4|zinoviev_alexander|0.2612|3|
|3822B1FI1|solovev_alexey|0.2613|1|
|3822B1PE1|korablev_vladlen|0.2780|3|
|3822B1PE1|rams_sergei|0.3064|1|
|3822B1PE1|gnitienko_kirill|0.3485|9|
|3822B1FI1|ionova_ekaterina|0.4071|5|
|3822B1PE1|khasanyanov_kirill|0.4413|13|
|3822B1PE1|odintsov_misha|0.4525|12|
|3822B1PE1|morozov_egor|0.5286|17|
|**REF**|**REF**|**0.5798**|**-**|
|3822B1PE2|titov_semyon|0.6015|1|
|3822B1PE3|sarafanov_maxim|1.0130|4|
|3822B1PE2|sorokin_andrey|1.0225|6|

## 5_block_gemm_omp (1024 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1PE2|mukhina_margarita|0.0221|3|
|**FAST**|**FAST**|**0.0231**|**-**|
|3822B1PE2|muradov_mike|0.0265|7|
|3822B1FI3|kirill_kholin|0.0292|3|
|3822B1FI3|kholin_kirill|0.0294|1|
|3822B1PE1|ermilova_darya|0.0297|15|
|3822B1PE2|kondratev_yaroslav|0.0302|5|
|3822B1PE4|shuravina_oksana|0.0374|2|
|3822B1PE1|milovankin_maxim|0.0404|8|
|3822B1PE4|zinoviev_alexander|0.0411|3|
|3822B1FI2|yasakova_tanya|0.0424|1|
|3822B1PE1|gnitienko_kirill|0.0464|9|
|3822B1FI1|komshina_daria|0.0918|4|
|3822B1PE3|oturin_alexander|0.0943|1|
|3822B1PE2|sorokin_andrey|0.0950|6|
|3822B1PE3|kazunin_nikita|0.0958|3|
|3822B1PE2|korovin_nikita|0.0959|4|
|3822B1PE2|titov_semyon|0.0988|1|
|3822B1PE1|khasanyanov_kirill|0.1015|14|
|3822B1PE1|sadikov_ivan|0.1235|5|
|3822B1PE1|krylov_mikhail|0.1379|7|
|3822B1PE1|rams_sergei|0.1406|3|
|3822B1PE4|kolokolova_darya|0.1430|4|
|3822B1PE1|nikolaev_roman|0.1434|11|
|3822B1PE1|konstantinov_ilya|0.1458|6|
|3822B1PE4|karaseva_ekaterina|0.1459|5|
|3822B1PE1|tyurin_mikhail|0.1463|10|
|3822B1PE2|filatieva_elizaveta|0.1464|8|
|3822B1PE2|filatiev_vladislav|0.1557|9|
|3822B1PE1|vershinina_alexandra|0.1618|13|
|**REF**|**REF**|**0.1670**|**-**|
|3822B1PE3|sarafanov_maxim|0.1730|4|
|3822B1FI3|lavrentyev_alexey|0.1733|4|
|3822B1FI1|shulpin_ilya|0.1761|2|
|3822B1PE1|moiseev_artem|0.1767|2|
|3822B1FI1|ionova_ekaterina|0.1794|5|
|3822B1PE1|shvedova_vitalina|0.1814|4|
|3822B1PE1|kalyakina_anastasia|0.1833|16|
|3822B1FI3|koshkin_nikita|0.1906|5|
|3822B1PE1|korablev_vladlen|0.1959|1|
|3822B1FI3|kudryashova_irina|0.2012|2|
|3822B1PE2|ermolaev_vladislav|0.2532|2|
|3822B1PE4|podovinnikov_artyom|0.2540|1|
|3822B1PE3|sotskov_andrey|0.2963|2|
|3822B1FI1|solovev_alexey|0.3512|1|
|3822B1FI1|elvin_veliev|0.3520|3|
|3822B1PE1|odintsov_misha|0.7827|12|
|3822B1PE1|morozov_egor|BUILD FAILED|-|

## 6_block_gemm_cuda (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1PE1|gnitienko_kirill|0.0377|7|
|**FAST**|**FAST**|**0.0776**|**-**|
|3822B1PE3|oturin_alexander|0.1036|3|
|3822B1PE1|krylov_mikhail|0.1113|3|
|3822B1PE3|sarafanov_maxim|0.1286|4|
|3822B1PE1|morozov_egor|0.1306|17|
|3822B1PE1|nikolaev_roman|0.1318|6|
|3822B1PE1|shvedova_vitalina|0.1390|10|
|3822B1FI2|yasakova_tanya|0.1396|1|
|3822B1FI3|kholin_kirill|0.1397|1|
|3822B1PE1|odintsov_misha|0.1404|11|
|3822B1PE4|zinoviev_alexander|0.1404|3|
|3822B1PE4|kolokolova_darya|0.1412|4|
|3822B1PE2|titov_semyon|0.1417|1|
|3822B1FI1|solovev_alexey|0.1424|1|
|3822B1FI3|kirill_kholin|0.1428|3|
|3822B1PE2|sorokin_andrey|0.1449|6|
|3822B1FI3|lavrentyev_alexey|0.1473|4|
|3822B1PE4|karaseva_ekaterina|0.1478|5|
|3822B1PE4|podovinnikov_artyom|0.1486|1|
|3822B1FI1|elvin_veliev|0.1489|3|
|3822B1PE2|kondratev_yaroslav|0.1504|5|
|3822B1PE2|filatiev_vladislav|0.1505|9|
|3822B1PE1|kalyakina_anastasia|0.1522|15|
|3822B1PE1|konstantinov_ilya|0.1544|4|
|3822B1PE1|khasanyanov_kirill|0.1601|12|
|3822B1PE2|filatieva_elizaveta|0.1776|8|
|3822B1PE2|mukhina_margarita|0.1836|3|
|3822B1FI1|shulpin_ilya|0.1843|2|
|3822B1FI1|komshina_daria|0.1861|4|
|3822B1FI3|kudryashova_irina|0.1866|2|
|3822B1PE1|milovankin_maxim|0.1871|5|
|3822B1PE1|rams_sergei|0.1877|1|
|3822B1PE1|moiseev_artem|0.1888|2|
|3822B1FI1|ionova_ekaterina|0.1912|5|
|3822B1PE1|korablev_vladlen|0.1917|9|
|3822B1PE1|sadikov_ivan|0.1932|8|
|3822B1PE3|sotskov_andrey|0.1993|1|
|3822B1PE2|ermolaev_vladislav|0.2001|2|
|3822B1PE1|tyurin_mikhail|0.2108|16|
|3822B1PE3|kazunin_nikita|0.2142|2|
|3822B1PE1|vershinina_alexandra|0.2497|14|
|3822B1PE2|muradov_mike|0.2517|7|
|**REF**|**REF**|**0.3026**|**-**|
|3822B1PE1|ermilova_darya|0.3264|13|
|3822B1FI3|koshkin_nikita|0.3405|5|
|3822B1PE2|korovin_nikita|0.3646|4|
|3822B1PE4|shuravina_oksana|0.4180|2|

## 7_gemm_cublas (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1PE1|tyurin_mikhail|0.0431|15|
|3822B1PE1|rams_sergei|0.0432|6|
|3822B1PE3|sarafanov_maxim|0.0432|4|
|3822B1PE1|milovankin_maxim|0.0438|4|
|3822B1PE2|korovin_nikita|0.0445|5|
|3822B1PE1|moiseev_artem|0.0446|1|
|3822B1FI1|solovev_alexey|0.0447|1|
|**FAST**|**FAST**|**0.0453**|**-**|
|3822B1PE1|korablev_vladlen|0.0455|7|
|3822B1PE1|shvedova_vitalina|0.0461|9|
|3822B1FI1|elvin_veliev|0.0465|3|
|3822B1PE1|konstantinov_ilya|0.0468|3|
|3822B1PE1|gnitienko_kirill|0.0480|10|
|3822B1PE2|ermolaev_vladislav|0.0499|2|
|3822B1PE2|titov_semyon|0.0503|1|
|3822B1PE1|sadikov_ivan|0.0515|5|
|3822B1PE4|shuravina_oksana|0.0517|2|
|3822B1FI3|lavrentyev_alexey|0.0518|3|
|3822B1PE3|sotskov_andrey|0.0524|1|
|3822B1FI1|ionova_ekaterina|0.0527|4|
|3822B1PE4|karaseva_ekaterina|0.0528|4|
|3822B1FI3|kirill_kholin|0.0529|2|
|3822B1PE1|kalyakina_anastasia|0.0530|16|
|3822B1PE2|sorokin_andrey|0.0531|4|
|3822B1PE4|kolokolova_darya|0.0531|5|
|3822B1FI3|koshkin_nikita|0.0531|4|
|3822B1FI2|yasakova_tanya|0.0532|1|
|3822B1PE2|kondratev_yaroslav|0.0535|3|
|3822B1PE4|podovinnikov_artyom|0.0535|1|
|3822B1FI1|shulpin_ilya|0.0538|2|
|3822B1PE1|krylov_mikhail|0.0547|2|
|3822B1PE3|kazunin_nikita|0.0547|2|
|3822B1PE1|khasanyanov_kirill|0.0562|11|
|**REF**|**REF**|**0.0563**|**-**|
|3822B1FI3|kudryashova_irina|0.0589|1|
|3822B1PE2|muradov_mike|0.0590|6|
|3822B1PE1|ermilova_darya|0.0681|13|
|3822B1FI1|komshina_daria|0.0763|5|
|3822B1PE3|oturin_alexander|0.0784|3|
|3822B1PE4|zinoviev_alexander|0.0787|3|
|3822B1PE2|mukhina_margarita|0.0789|7|
|3822B1PE1|odintsov_misha|0.1022|12|
|3822B1PE1|vershinina_alexandra|0.1218|14|
|3822B1PE2|filatiev_vladislav|0.5419|9|
|3822B1PE2|filatieva_elizaveta|0.7555|8|
|3822B1PE1|nikolaev_roman|0.7786|8|
|3822B1PE1|morozov_egor|TEST FAILED|-|
|3822B1FI3|kholin_kirill|TEST FAILED|-|

## 8_fft_cufft (131072 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|**FAST**|**FAST**|**0.1075**|**-**|
|3822B1PE1|rams_sergei|0.1094|4|
|3822B1PE1|moiseev_artem|0.1100|1|
|3822B1PE1|ermilova_darya|0.1105|13|
|3822B1PE1|tyurin_mikhail|0.1108|14|
|3822B1PE1|krylov_mikhail|0.1109|2|
|3822B1PE1|shvedova_vitalina|0.1117|9|
|3822B1PE3|sarafanov_maxim|0.1163|4|
|3822B1PE1|milovankin_maxim|0.1222|5|
|3822B1PE1|khasanyanov_kirill|0.1226|12|
|3822B1PE4|karaseva_ekaterina|0.1226|5|
|3822B1PE2|korovin_nikita|0.1231|3|
|3822B1PE4|podovinnikov_artyom|0.1235|1|
|3822B1FI3|lavrentyev_alexey|0.1235|4|
|3822B1FI3|koshkin_nikita|0.1240|3|
|3822B1FI1|ionova_ekaterina|0.1245|4|
|3822B1PE3|oturin_alexander|0.1248|2|
|3822B1PE1|gnitienko_kirill|0.1249|16|
|3822B1PE4|kolokolova_darya|0.1266|4|
|3822B1PE2|ermolaev_vladislav|0.1272|2|
|3822B1PE3|kazunin_nikita|0.1279|1|
|3822B1PE4|shuravina_oksana|0.1280|2|
|3822B1PE1|odintsov_misha|0.1291|11|
|3822B1PE1|sadikov_ivan|0.1300|6|
|3822B1FI1|solovev_alexey|0.1304|1|
|3822B1FI1|elvin_veliev|0.1309|3|
|3822B1FI3|kirill_kholin|0.1311|1|
|3822B1PE2|filatiev_vladislav|0.1347|9|
|3822B1PE1|nikolaev_roman|0.1365|7|
|3822B1PE2|mukhina_margarita|0.1397|5|
|3822B1FI1|shulpin_ilya|0.1400|2|
|3822B1PE1|korablev_vladlen|0.1401|8|
|3822B1PE1|kalyakina_anastasia|0.1406|15|
|3822B1FI1|komshina_daria|0.1420|5|
|3822B1PE2|muradov_mike|0.1422|4|
|3822B1PE3|sotskov_andrey|0.1424|3|
|3822B1PE2|kondratev_yaroslav|0.1438|7|
|3822B1FI2|yasakova_tanya|0.1459|1|
|3822B1PE2|sorokin_andrey|0.1543|6|
|3822B1FI3|kudryashova_irina|0.1546|2|
|3822B1PE2|filatieva_elizaveta|0.1555|8|
|3822B1PE2|titov_semyon|0.1697|1|
|3822B1PE4|zinoviev_alexander|0.1713|3|
|**REF**|**REF**|**0.2228**|**-**|
|3822B1PE1|vershinina_alexandra|0.2334|10|
|3822B1PE1|konstantinov_ilya|0.2411|3|

## 9_gelu_ocl (134217728 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|**FAST**|**FAST**|**0.1188**|**-**|
|3822B1PE4|karaseva_ekaterina|0.2239|3|
|3822B1PE2|korovin_nikita|0.2294|5|
|3822B1FI1|ionova_ekaterina|0.2464|5|
|3822B1PE3|oturin_alexander|0.2614|3|
|3822B1FI3|kirill_kholin|0.2672|2|
|3822B1FI3|lavrentyev_alexey|0.2706|3|
|3822B1PE1|khasanyanov_kirill|0.2711|10|
|3822B1PE3|sotskov_andrey|0.2949|2|
|3822B1FI3|kudryashova_irina|0.2983|1|
|3822B1PE4|kolokolova_darya|0.2995|4|
|3822B1FI1|shulpin_ilya|0.3014|1|
|3822B1PE1|krylov_mikhail|0.3019|2|
|3822B1PE2|ermolaev_vladislav|0.3052|7|
|3822B1PE1|rams_sergei|0.3110|5|
|3822B1PE4|podovinnikov_artyom|0.3114|1|
|3822B1PE1|moiseev_artem|0.3126|1|
|3822B1PE1|korablev_vladlen|0.3132|7|
|3822B1PE4|shuravina_oksana|0.3139|2|
|3822B1PE3|kazunin_nikita|0.3168|1|
|3822B1PE1|milovankin_maxim|0.3182|3|
|3822B1PE2|filatiev_vladislav|0.3362|9|
|3822B1PE2|muradov_mike|0.3372|8|
|3822B1FI1|elvin_veliev|0.3385|3|
|3822B1PE1|ermilova_darya|0.3401|11|
|3822B1PE1|shvedova_vitalina|0.3413|8|
|**REF**|**REF**|**0.3419**|**-**|
|3822B1FI1|solovev_alexey|0.3438|2|
|3822B1FI1|komshina_daria|0.3439|4|
|3822B1PE1|odintsov_misha|0.3453|12|
|3822B1PE1|tyurin_mikhail|0.3455|13|
|3822B1PE1|sadikov_ivan|0.3460|14|
|3822B1PE2|filatieva_elizaveta|0.3476|4|
|3822B1PE2|mukhina_margarita|0.3493|2|
|3822B1PE2|kondratev_yaroslav|0.3494|6|
|3822B1PE2|titov_semyon|0.3591|1|
|3822B1PE1|nikolaev_roman|0.3601|6|
|3822B1PE1|konstantinov_ilya|0.3629|4|
|3822B1PE2|sorokin_andrey|0.3648|3|
|3822B1PE1|vershinina_alexandra|0.6035|9|
|3822B1PE4|zinoviev_alexander|RUN FAILED|-|
|3822B1PE3|sarafanov_maxim|RUN FAILED|-|
|3822B1FI3|koshkin_nikita|BUILD FAILED|-|
|3822B1FI2|yasakova_tanya|BUILD FAILED|-|

# Tasks Done
## 3822B1FI1
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3822B1FI1|Ionova_ekaterina|1/9|60|
|3822B1FI1|elvin_veliev|**9/9**|**540**|
|3822B1FI1|ionova_ekaterina|**9/9**|**527**|
|3822B1FI1|komshina_daria|**9/9**|**522**|
|3822B1FI1|shulpin_ilya|**9/9**|**548**|
|3822B1FI1|solovev_alexey|**9/9**|**557**|

Passed: 5

## 3822B1FI2
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3822B1FI2|sdobnov_vladimir|0/9|0|
|3822B1FI2|yasakova_tanya|8/9|512|

Passed: 0

## 3822B1FI3
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3822B1FI3|kholin_kirill|3/9|190|
|3822B1FI3|kirill_kholin|**9/9**|**555**|
|3822B1FI3|koshkin_nikita|8/9|468|
|3822B1FI3|kudryashova_irina|**9/9**|**548**|
|3822B1FI3|lavrentyev_alexey|**9/9**|**537**|
|3822B1FI3|solovyev_danila|2/9|118|

Passed: 3

## 3822B1PE1
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3822B1PE1|ermilova_darya|**9/9**|**403**|
|3822B1PE1|gnitienko_kirill|8/9|378|
|3822B1PE1|kalyakina_anastasia|8/9|326|
|3822B1PE1|khasanyanov_kirill|**9/9**|**400**|
|3822B1PE1|konstantinov_ilya|**9/9**|**462**|
|3822B1PE1|korablev_vladlen|**9/9**|**467**|
|3822B1PE1|krylov_mikhail|**9/9**|**485**|
|3822B1PE1|milovankin_maxim|**9/9**|**469**|
|3822B1PE1|moiseev_artem|**9/9**|**509**|
|3822B1PE1|morozov_egor|5/9|223|
|3822B1PE1|nikolaev_roman|**9/9**|**416**|
|3822B1PE1|odintsov_misha|**9/9**|**369**|
|3822B1PE1|polikanov_vitaliy|2/9|88|
|3822B1PE1|rams_sergei|**9/9**|**505**|
|3822B1PE1|sadikov_ivan|**9/9**|**444**|
|3822B1PE1|shvedova_vitalina|**9/9**|**465**|
|3822B1PE1|sidorina_polina|1/9|39|
|3822B1PE1|tyurin_mikhail|**9/9**|**434**|
|3822B1PE1|vershinina_alexandra|**9/9**|**352**|

Passed: 14

## 3822B1PE2
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3822B1PE2|ermolaev_vladislav|**9/9**|**533**|
|3822B1PE2|filatiev_vladislav|**9/9**|**471**|
|3822B1PE2|filatieva_elizaveta|**9/9**|**472**|
|3822B1PE2|kondratev_yaroslav|**9/9**|**504**|
|3822B1PE2|korovin_nikita|**9/9**|**522**|
|3822B1PE2|mukhina_margarita|**9/9**|**515**|
|3822B1PE2|muradov_mike|**9/9**|**494**|
|3822B1PE2|sorokin_andrey|**9/9**|**491**|
|3822B1PE2|titov_semyon|**9/9**|**534**|
|3822B1PE2|vladimirova_julia|0/9|0|

Passed: 9

## 3822B1PE3
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3822B1PE3|kazunin_nikita|**9/9**|**548**|
|3822B1PE3|oturin_alexander|**9/9**|**559**|
|3822B1PE3|sarafanov_maxim|7/9|416|
|3822B1PE3|sotskov_andrey|**9/9**|**557**|

Passed: 3

## 3822B1PE4
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3822B1PE4|karaseva_ekaterina|**9/9**|**528**|
|3822B1PE4|kolokolova_darya|**9/9**|**533**|
|3822B1PE4|podovinnikov_artyom|**9/9**|**555**|
|3822B1PE4|shuravina_oksana|**9/9**|**550**|
|3822B1PE4|zinoviev_alexander|8/9|478|

Passed: 4

**Total Passed: 38**

---
*Maximum Score: 576 (64 per task)
*