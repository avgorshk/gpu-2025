#define CL_TARGET_OPENCL_VERSION 220
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <string>

const std::string GELU_KERNEL = R"(
__kernel void geluKernel(__global const float* input, __global float* output, int n) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    
    int idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coef * x3);
        output[idx] = 0.5f * x * (1.0f + tanh(inner));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t n = input.size();
    std::vector<float> output(n);

    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    
    cl_platform_id platformId = platforms[platform];

    cl_device_id deviceId;
    clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, deviceId, NULL, NULL);

    const char* sourceStr = GELU_KERNEL.c_str();
    size_t sourceSize = GELU_KERNEL.size();
    cl_program program = clCreateProgramWithSource(context, 1, &sourceStr, &sourceSize, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "geluKernel", nullptr);

    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        n * sizeof(float), (void*)input.data(), nullptr);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         n * sizeof(float), nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, n * sizeof(float), output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(inputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}