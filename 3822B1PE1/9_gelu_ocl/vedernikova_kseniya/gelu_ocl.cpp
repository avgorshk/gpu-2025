#include "gelu_ocl.h"
#include <CL/opencl.h>
#include <cmath>
#include <string>

const char *kernelSource = R"(
__global__ void gelu_kernel(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    const float c = 0.044715f;
    const float a = 0.79788456f; 
    if (idx < n) {
        float x = input[idx];
        float tanh_arg = a * x * (1.0f + c * x * x);
        float tanh = (exp(tanh_arg) - (1 / exp(tanh_arg))) / (exp(tanh_arg) + (1 / exp(tanh_arg)));
        output[idx] = 0.5f * x * (1.0f + tanh);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float> &input, int platform) {
    int n = (int)input.size();

    std::vector<float> result(n);

    cl_platform_id platformId;
    cl_device_id device;
    cl_uint countPlatforms, numDevices;

    clGetPlatformIDs(0, nullptr, &countPlatforms);
    cl_platform_id *platforms = new cl_platform_id[countPlatforms];
    clGetPlatformIDs(countPlatforms, platforms, nullptr);
    platformId = platforms[platform];
    delete[] platforms;

    clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);

    cl_int error;
    cl_context context =
            clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);

    size_t mem = n * sizeof(float);
    cl_mem d_input =
            clCreateBuffer(context, CL_MEM_READ_ONLY, mem, nullptr, &error);
    cl_mem d_output =
            clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem, nullptr, &error);

    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, mem, input.data(), 0,
                         nullptr, nullptr);

    cl_program program =
            clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &error);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "gelu", &error);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t size1 = (n + 255) / 256 * 256;
    size_t size2 = 256;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &size1, &size2, 0,
                           nullptr, nullptr);

    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, mem, result.data(), 0,
                        nullptr, nullptr);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return result;
}
