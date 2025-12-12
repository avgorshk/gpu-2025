
#define CL_TARGET_OPENCL_VERSION 300

#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <iostream>

static const char* gelu_kernel_code = R"CLC(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const unsigned int n) 
{
    const size_t idx = get_global_id(0);

    if (idx < n) {
        float x = input[idx];
        float tanh_result = tanh(sqrt(2.0f / 3.14159265f) * x * (1.0f + 0.044715f * x * x));
        output[idx] = 0.5f * x * (1.0f + tanh_result);
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    std::vector<float> output(input.size());
    cl_int err;

    cl_uint num_platforms;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_platform_id aplatform = platforms[platform];

    cl_uint num_devices;
    clGetDeviceIDs(aplatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(aplatform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    const char* source = gelu_kernel_code;
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);

    cl_mem buf_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   input.size() * sizeof(float), (void*)input.data(), &err);
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    output.size() * sizeof(float), nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    unsigned int n = static_cast<unsigned int>(input.size());
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &n);

    size_t global_work_size = input.size();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, output.size() * sizeof(float), output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
