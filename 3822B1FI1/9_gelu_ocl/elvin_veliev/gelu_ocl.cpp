#include "gelu_ocl.h"

#include <CL/cl.h>
#include <cmath>
#include <vector>

const char* kernelSource = R"CLC(
__kernel void gelu_kernel(__global const float* in,
                          __global float* out,
                          const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        const float x = in[i];
        const float c = 0.044715f;
        const float sqrt_2_over_pi = 0.7978845608028654f;

        float x3 = x * x * x;
        float z = sqrt_2_over_pi * (x + c * x3);
        float s = 1.0f / (1.0f + exp(-2.0f * z));
        out[i] = x * s;
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    cl_platform_id platform = platforms[platform];

    cl_uint numDevices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);

    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    cl_device_id device = devices[0];

    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);

    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel_cl = clCreateKernel(program, "gelu_kernel", &err);

    const size_t bytes = input.size() * sizeof(float);
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)input.data(), &err);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    int n = static_cast<int>(input.size());
    clSetKernelArg(kernel_cl, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel_cl, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel_cl, 2, sizeof(int), &n);

    size_t globalSize = ((n + 255) / 256) * 256;
    size_t localSize = 256;
    clEnqueueNDRangeKernel(queue, kernel_cl, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);

    std::vector<float> output(input.size());
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel_cl);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}