#include "gelu_ocl.h"

#include <CL/cl.h>
#include <cmath>

const char* geluKernelSource = R"(
__kernel void gelu(__global const float* input,
                   __global float* output,
                   const int size)
{
    int id = get_global_id(0);
    if (id < size) {
        float x = input[id];
        float t = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[id] = 0.5f * x * (1.0f + tanh(t));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    const int size = static_cast<int>(input.size());
    const size_t bytes = size * sizeof(float);

    cl_platform_id platform_id;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem inputBuffer, outputBuffer;

    clGetPlatformIDs(1, &platform_id, nullptr);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue   = clCreateCommandQueue(context, device, 0, nullptr);

    program = clCreateProgramWithSource(context, 1, &geluKernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);

    kernel = clCreateKernel(program, "gelu", nullptr);

    inputBuffer  = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                                  bytes, (void*)input.data(), nullptr);
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  bytes, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &size);

    const size_t localSize  = 256;
    const size_t globalSize = ((size + localSize - 1) / localSize) * localSize;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                           &globalSize, &localSize,
                           0, nullptr, nullptr);

    std::vector<float> output(size);
    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE,
                        0, bytes, output.data(),
                        0, nullptr, nullptr);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}