#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>

const char* kernel_src = R"(
__kernel void gelu(__global const float* __restrict__ input,
                   __global float* __restrict__ output,
                   int n) {
    const int idx = get_global_id(0);
    if (idx < n) {
        const float x = input[idx];
        const float x3 = x * x * x;
        const float arg = -1.59576912f * (x + 0.044715f * x3);
        const float exp_arg = exp(arg);
        output[idx] = x / (1.0f + exp_arg);
    }
}
)";

static cl_context ctx = nullptr;
static cl_command_queue queue = nullptr;
static cl_kernel kernel = nullptr;
static int initialized_platform = -1;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const int n = input.size();

    if (initialized_platform != platform) {
        initialized_platform = platform;

        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        cl_platform_id pl = platforms[platform];

        cl_uint num_devices;
        clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

        cl_int err;
        ctx = clCreateContext(nullptr, 1, &devices[0], nullptr, nullptr, &err);
        queue = clCreateCommandQueueWithProperties(ctx, devices[0], 0, &err);

        cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_src, nullptr, &err);
        clBuildProgram(program, 1, &devices[0], nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu", &err);

        clReleaseProgram(program);
    }

    cl_int err;
    cl_mem input_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      n * sizeof(float), const_cast<float*>(input.data()), &err);
    cl_mem output_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    const size_t local_size = 256;
    const size_t global_size = ((n + local_size - 1) / local_size) * local_size;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);

    std::vector<float> output(n);
    clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, n * sizeof(float), output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);

    return output;
}
