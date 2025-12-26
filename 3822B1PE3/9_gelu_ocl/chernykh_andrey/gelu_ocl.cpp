#include "gelu_ocl.h"
#include <vector>
#include <cmath>
#include <CL/cl.h>

const char *kernelSource = R"(
__kernel void gelu_kernel(
    __global const float *input,
    __global float *output,
    int n
) {
    int id = get_global_id(0);
    if (id >= n) {
        return;
    }

    const float SQRT_2_PI = 0.7978845608f;
    const float COEF = 0.044715f;

    float x = input[id];
    float arg = SQRT_2_PI * (x + COEF * x * x * x);
    output[id] = 0.5f * x * (1.0f + tanh(arg));
}
)";

static cl_context context = nullptr;
static cl_command_queue queue = nullptr;
static cl_program program = nullptr;
static cl_kernel kernel = nullptr;
static bool is_initialized = false;

void InitOpenCL(int target_platform_index) {
    if (is_initialized) {
        return;
    }

    cl_int err;

    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    if (target_platform_index < 0 || target_platform_index >= num_platforms) {
        return;
    }
    cl_platform_id platform = platforms[target_platform_index];

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        return;
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "gelu_kernel", &err);

    is_initialized = true;
}

std::vector<float> GeluOCL(const std::vector<float> &input, int platform) {
    if (input.empty()) {
        return {};
    }

    InitOpenCL(platform);
    if (!is_initialized) {
        return {};
    }

    cl_int err;
    auto size = static_cast<int>(input.size());
    size_t data_size = size * sizeof(float);

    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &err);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, &err);
    err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, data_size, input.data(), 0, NULL, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &size);

    size_t local_size = 256;
    size_t global_size = (size + local_size - 1) / local_size * local_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    std::vector<float> output(size);
    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, data_size, output.data(), 0, NULL, NULL);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    return output;
}
