#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <stdexcept>
#include <string>

static const std::string KERNEL_CODE = R"(
__kernel void compute_func(__global const float* data_in, __global float* data_out, int total) {
    const float factor_a = 0.7978845608028654f;
    const float factor_b = 0.044715f;
    
    int position = get_global_id(0);
    if (position < total) {
        float value = data_in[position];
        float cubed = value * value * value;
        float computation = factor_a * (value + factor_b * cubed);
        data_out[position] = 0.5f * value * (1.0f + tanh(computation));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    std::vector<float> result;
    if (input.empty()) return result;

    cl_uint platform_count;
    clGetPlatformIDs(0, NULL, &platform_count);

    std::vector<cl_platform_id> platforms_list(platform_count);
    clGetPlatformIDs(platform_count, platforms_list.data(), NULL);

    cl_platform_id selected_platform = platforms_list[platform];
    cl_device_id target_device;
    clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 1, &target_device, NULL);

    cl_context context = clCreateContext(NULL, 1, &target_device, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, target_device, 0, NULL);

    const char* kernel_source = KERNEL_CODE.c_str();
    size_t source_length = KERNEL_CODE.size();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, &source_length, NULL);
    clBuildProgram(program, 1, &target_device, NULL, NULL, NULL);

    cl_kernel kernel_func = clCreateKernel(program, "compute_func", NULL);

    size_t element_count = input.size();
    cl_mem input_memory = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        element_count * sizeof(float), (void*)input.data(), NULL);
    cl_mem output_memory = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        element_count * sizeof(float), NULL, NULL);

    clSetKernelArg(kernel_func, 0, sizeof(cl_mem), &input_memory);
    clSetKernelArg(kernel_func, 1, sizeof(cl_mem), &output_memory);
    clSetKernelArg(kernel_func, 2, sizeof(int), &element_count);

    size_t work_size = element_count;
    clEnqueueNDRangeKernel(command_queue, kernel_func, 1, NULL, &work_size, NULL, 0, NULL, NULL);

    result.resize(element_count);
    clEnqueueReadBuffer(command_queue, output_memory, CL_TRUE, 0,
        element_count * sizeof(float), result.data(), 0, NULL, NULL);

    clReleaseMemObject(output_memory);
    clReleaseMemObject(input_memory);
    clReleaseKernel(kernel_func);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return result;
}