#include "gelu_ocl.h"
#include <CL/cl.h>
#include <stdexcept>
#include <cmath>

static const char* gelu_kernel_source = 
"__kernel void gelu(__global const float* input, __global float* output, const int n) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx < n) {\n"
"        float x = input[idx];\n"
"        float x3 = x * x * x;\n"
"        float inner = 0.7978845608f * (x + 0.044715f * x3);\n"
"        float exp_val = exp(inner);\n"
"        float exp_neg = 1.0f / exp_val;\n"
"        float tanh_val = (exp_val - exp_neg) / (exp_val + exp_neg);\n"
"        output[idx] = 0.5f * x * (1.0f + tanh_val);\n"
"    }\n"
"}\n";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get platform IDs");
    }
    
    if (platform < 0 || static_cast<size_t>(platform) >= num_platforms) {
        throw std::runtime_error("Invalid platform index");
    }
    
    cl_platform_id selected_platform = platforms[platform];
    
    cl_uint num_devices;
    err = clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        throw std::runtime_error("No GPU devices found");
    }
    
    cl_device_id device;
    err = clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get device ID");
    }
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create context");
    }
    
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        throw std::runtime_error("Failed to create command queue");
    }
    
    cl_program program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create program");
    }
    
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to build program");
    }
    
    cl_kernel kernel = clCreateKernel(program, "gelu", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create kernel");
    }
    
    size_t data_size = input.size() * sizeof(float);
    
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create input buffer");
    }
    
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create output buffer");
    }
    
    err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, data_size, input.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to write input buffer");
    }
    
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &input.size());
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to set kernel arguments");
    }
    
    size_t global_size = input.size();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to enqueue kernel");
    }
    
    std::vector<float> output(input.size());
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, data_size, output.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to read output buffer");
    }
    
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(input_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return output;
}

