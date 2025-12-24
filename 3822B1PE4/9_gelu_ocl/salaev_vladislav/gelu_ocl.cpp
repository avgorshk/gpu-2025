#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "gelu_ocl.h"
#include <CL/cl.h>
#include <string>
#include <cstring>
#include <vector>

const char *kernelSource = R"(
__kernel void gelu(__global const float* input, 
                   __global float* output, 
                   int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        float exp2x = exp(2.0f * inner);
        float tanh_val = (exp2x - 1.0f) / (exp2x + 1.0f);
        
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}
)";

static cl_context context = nullptr;
static cl_command_queue queue = nullptr;
static cl_program program = nullptr;
static cl_kernel kernel = nullptr;
static bool initialized = false;
static int cached_platform = -1;

void initOpenCL(int platform)
{
    if (initialized && cached_platform == platform)
        return;

    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    cl_device_id device;
    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, device, 0, nullptr);

    size_t sourceSize = strlen(kernelSource);
    program = clCreateProgramWithSource(context, 1, &kernelSource, &sourceSize, nullptr);
    clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);

    kernel = clCreateKernel(program, "gelu", nullptr);

    initialized = true;
    cached_platform = platform;
}

std::vector<float> GeluOCL(const std::vector<float> &input, int platform)
{
    initOpenCL(platform);

    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    bytes, (void *)input.data(), nullptr);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    int n_int = static_cast<int>(n);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n_int);

    size_t globalSize = ((n + 255) / 256) * 256;
    size_t localSize = 256;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                           0, nullptr, nullptr);

    std::vector<float> output(n);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(),
                        0, nullptr, nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);

    return output;
}