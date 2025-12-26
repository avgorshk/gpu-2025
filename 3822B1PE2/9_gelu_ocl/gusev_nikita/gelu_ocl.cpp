#include "gelu_ocl.h"

#include <CL/cl.h>
#include <cstdint>
#include <cstring>

namespace {

const char* kKernelSource = R"(
__kernel void gelu(__global const float* input, __global float* output, int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float x3 = x * x * x;
        float z = 0.7978845608f * (x + 0.044715f * x3);
        float e = exp(2.0f * z);
        float t = (e - 1.0f) / (e + 1.0f);
        output[i] = 0.5f * x * (1.0f + t);
    }
}
)";

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t n = input.size();
    std::vector<float> output(n);

    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    cl_platform_id selectedPlatform = platforms[platform];

    cl_device_id device;
    clGetDeviceIDs(selectedPlatform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    size_t srcLen = strlen(kKernelSource);
    cl_program program = clCreateProgramWithSource(context, 1, &kKernelSource, &srcLen, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "gelu", nullptr);

    size_t bufSize = n * sizeof(float);
    cl_mem inputBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, bufSize, nullptr, nullptr);
    cl_mem outputBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufSize, nullptr, nullptr);

    clEnqueueWriteBuffer(queue, inputBuf, CL_TRUE, 0, bufSize, input.data(), 0, nullptr, nullptr);

    int nInt = static_cast<int>(n);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuf);
    clSetKernelArg(kernel, 2, sizeof(int), &nInt);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, outputBuf, CL_TRUE, 0, bufSize, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputBuf);
    clReleaseMemObject(outputBuf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}

