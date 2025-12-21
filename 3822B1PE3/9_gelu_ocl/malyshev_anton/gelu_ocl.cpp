#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>

const char* kernelSource = R"(
__kernel void gelu(__global const float* __restrict__ input,
                   __global float* __restrict__ output,
                   int size) {
    const int i = get_global_id(0);
    if (i < size) {
        const float x = input[i];
        const float x3 = x * x * x;
        const float arg = -1.59576912f * (x + 0.044715f * x3);
        const float expArg = exp(arg);
        output[i] = x / (1.0f + expArg);
    }
}
)";

static cl_context context = nullptr;
static cl_command_queue commandQueue = nullptr;
static cl_kernel oclKernel = nullptr;
static int initializedPlatform = -1;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const int size = input.size();

    if (initializedPlatform != platform) {
        initializedPlatform = platform;

        cl_uint numPlatforms;
        clGetPlatformIDs(0, nullptr, &numPlatforms);

        std::vector<cl_platform_id> platforms(numPlatforms);
        clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        cl_platform_id selectedPlatform = platforms[platform];

        cl_uint numDevices;
        clGetDeviceIDs(selectedPlatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);

        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(selectedPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);

        cl_int errorCode;
        context = clCreateContext(nullptr, 1, &devices[0], nullptr, nullptr, &errorCode);
        commandQueue = clCreateCommandQueueWithProperties(context, devices[0], 0, &errorCode);

        cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &errorCode);
        clBuildProgram(program, 1, &devices[0], nullptr, nullptr, nullptr);
        oclKernel = clCreateKernel(program, "gelu", &errorCode);

        clReleaseProgram(program);
    }

    cl_int errorCode;
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      size * sizeof(float), const_cast<float*>(input.data()), &errorCode);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), nullptr, &errorCode);

    clSetKernelArg(oclKernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(oclKernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(oclKernel, 2, sizeof(int), &size);

    const size_t localSize = 256;
    const size_t globalSize = ((size + localSize - 1) / localSize) * localSize;

    clEnqueueNDRangeKernel(commandQueue, oclKernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);

    std::vector<float> output(size);
    clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, size * sizeof(float), output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);

    return output;
}
