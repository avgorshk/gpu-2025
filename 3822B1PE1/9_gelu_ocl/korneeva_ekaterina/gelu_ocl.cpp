#include "gelu_ocl.h"

#include <CL/opencl.h>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

#define CHECK_CL_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        throw std::runtime_error(std::string(msg) + " (Error code: " + std::to_string(err) + ")"); \
    }

static cl_context context = nullptr;
static cl_command_queue command_queue = nullptr;
static cl_program program = nullptr;
static cl_kernel kernel = nullptr;
static cl_device_id device_id = nullptr;
static cl_mem d_input = nullptr;
static cl_mem d_output = nullptr;
static size_t allocated_size = 0;

const std::string kernel_source = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3);
        float exp_inner = exp(inner);
        float exp_neg_inner = exp(-inner);
        float tanh_value = (exp_inner - exp_neg_inner) / (exp_inner + exp_neg_inner);
        output[idx] = 0.5f * x * (1.0f + tanh_value);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t size = input.size();
    std::vector<float> output(size);

    if (size == 0) {
        return output;
    }

    cl_int err;

    if (context == nullptr) {
        cl_platform_id platforms[10];
        cl_uint num_platforms;
        err = clGetPlatformIDs(10, platforms, &num_platforms);
        CHECK_CL_ERROR(err, "Failed to get platforms");

        if (platform < 0 || static_cast<cl_uint>(platform) >= num_platforms) {
            throw std::runtime_error("Invalid platform index");
        }

        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
        CHECK_CL_ERROR(err, "Failed to get GPU device");

        context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
        CHECK_CL_ERROR(err, "Failed to create context");

        command_queue = clCreateCommandQueue(context, device_id, 0, &err);
        CHECK_CL_ERROR(err, "Failed to create command queue");

        const char* source_str = kernel_source.c_str();
        program = clCreateProgramWithSource(context, 1, &source_str, nullptr, &err);
        CHECK_CL_ERROR(err, "Failed to create program");

        err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> build_log(log_size);
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
            throw std::runtime_error("Failed to build program: " + std::string(build_log.data()));
        }

        kernel = clCreateKernel(program, "gelu_kernel", &err);
        CHECK_CL_ERROR(err, "Failed to create kernel");
    }

    size_t required_size = size * sizeof(float);

    if (allocated_size < required_size) {
        if (d_input) clReleaseMemObject(d_input);
        if (d_output) clReleaseMemObject(d_output);

        d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, required_size, nullptr, &err);
        CHECK_CL_ERROR(err, "Failed to create input buffer");

        d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, required_size, nullptr, &err);
        CHECK_CL_ERROR(err, "Failed to create output buffer");

        allocated_size = required_size;
    }

    err = clEnqueueWriteBuffer(command_queue, d_input, CL_TRUE, 0, required_size, input.data(), 0, nullptr, nullptr);
    CHECK_CL_ERROR(err, "Failed to write input buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    CHECK_CL_ERROR(err, "Failed to set kernel arg 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    CHECK_CL_ERROR(err, "Failed to set kernel arg 1");

    int size_arg = static_cast<int>(size);
    err = clSetKernelArg(kernel, 2, sizeof(int), &size_arg);
    CHECK_CL_ERROR(err, "Failed to set kernel arg 2");

    size_t global_work_size = size;
    size_t local_work_size = 256;
    global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;

    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    CHECK_CL_ERROR(err, "Failed to enqueue kernel");

    err = clFinish(command_queue);
    CHECK_CL_ERROR(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(command_queue, d_output, CL_TRUE, 0, required_size, output.data(), 0, nullptr, nullptr);
    CHECK_CL_ERROR(err, "Failed to read output buffer");

    return output;
}