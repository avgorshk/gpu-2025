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

std::vector<float> GeluOCL(const std::vector<float>& input, int platformIndex) {
    if (input.empty()) return {};

    cl_int err;
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    cl_platform_id platform = platforms[platformIndex];

    cl_uint numDevices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);

    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    cl_program program = clCreateProgramWithSource(context, 1, &geluKernelSource, nullptr, &err);

    cl_kernel kernel = clCreateKernel(program, "gelu", &err);

    const size_t bytes = input.size() * sizeof(float);
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)input.data(), &err);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    int n = static_cast<int>(input.size());
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t globalSize = ((n + 255) / 256) * 256;
    size_t localSize = 256;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);

    std::vector<float> output(input.size());
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}