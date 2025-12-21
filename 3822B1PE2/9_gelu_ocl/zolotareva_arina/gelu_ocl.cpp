#include "gelu_ocl.h"

#include <CL/cl.h>
#include <vector>
#include <cstring>

static const char* gelu_kernel_src = R"CLC(
__kernel void gelu(__global const float* input,
                   __global float* output,
                   int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        // exp-based GELU approximation
        float x3 = x * x * x;
        float t = 0.7978845608f * (x + 0.044715f * x3);
        float y = 1.0f / (1.0f + exp(-2.0f * t));
        output[i] = x * y;
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const int n = static_cast<int>(input.size());
    std::vector<float> output(n);

    cl_int err;

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_platform_id plat = platforms[platform];

    cl_device_id device;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    cl_mem d_input = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        n * sizeof(float),
        (void*)input.data(),
        &err
    );

    cl_mem d_output = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        n * sizeof(float),
        nullptr,
        &err
    );

    cl_program program = clCreateProgramWithSource(
        context, 1, &gelu_kernel_src, nullptr, &err
    );

    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "gelu", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t global = ((n + 255) / 256) * 256;
    size_t local = 256;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, n * sizeof(float), output.data(), 0, nullptr, nullptr);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
