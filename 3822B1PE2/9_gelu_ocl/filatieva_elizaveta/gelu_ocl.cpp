#include "gelu_ocl.h"
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <memory>

const std::string GELU_KERNEL_SOURCE = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
    const float scale_factor = 0.7978845608028654f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    
    int idx = get_global_id(0);
    if (idx < n) {
        float value = input[idx];
        float cube = value * value * value;
        float transformed = scale_factor * (value + coeff * cube);
        
        // Используем экспоненциальную формулу вместо tanh для лучшей производительности
        float exp_val = exp(2.0f * transformed);
        float tanh_approx = (exp_val - 1.0f) / (exp_val + 1.0f);
        
        output[idx] = 0.5f * value * (1.0f + tanh_approx);
    }
}
)";

struct OpenCLResources {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    OpenCLResources(cl_context c, cl_command_queue q, cl_program p, cl_kernel k)
        : context(c), queue(q), program(p), kernel(k) {}

    ~OpenCLResources() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return std::vector<float>();
    }

    cl_int err;
    cl_uint num_platforms;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms available");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to retrieve OpenCL platforms");
    }

    if (platform < 0 || platform >= static_cast<int>(num_platforms)) {
        throw std::runtime_error("Invalid platform index");
    }

    cl_platform_id selected_platform = platforms[platform];
    cl_device_id device_id;

    err = clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("No GPU device found on selected platform");
    }

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context");
    }

    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        throw std::runtime_error("Failed to create command queue");
    }

    const char* source_str = GELU_KERNEL_SOURCE.c_str();
    size_t source_size = GELU_KERNEL_SOURCE.size();
    cl_program program = clCreateProgramWithSource(context, 1, &source_str, &source_size, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create OpenCL program");
    }

    err = clBuildProgram(program, 1, &device_id, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);

        std::string error_msg = "OpenCL program build failed: " + std::string(build_log.data());

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error(error_msg);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create kernel");
    }

    auto resources = std::make_shared<OpenCLResources>(context, queue, program, kernel);

    size_t data_size = input.size();

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        data_size * sizeof(float), (void*)input.data(), &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create input buffer");
    }

    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        data_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        throw std::runtime_error("Failed to create output buffer");
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &data_size);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        throw std::runtime_error("Failed to set kernel arguments");
    }

    size_t max_work_group_size;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);

    size_t local_size = std::min(max_work_group_size, (size_t)256);
    size_t global_size = ((data_size + local_size - 1) / local_size) * local_size;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        throw std::runtime_error("Failed to execute kernel");
    }

    clFinish(queue);

    std::vector<float> output(data_size);
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
        data_size * sizeof(float), output.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        throw std::runtime_error("Failed to read output buffer");
    }

    clReleaseMemObject(output_buffer);
    clReleaseMemObject(input_buffer);

    return output;
}