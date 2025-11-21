#include <CL/cl.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error (" << err << ") during operation: " << operation << std::endl;
        exit(1); 
    }
}

const std::string gelu_kernel_code = R"(
    __kernel void gelu(__global float* input, __global float* output, const unsigned int size) {
        int i = get_global_id(0);
        if (i < size) {
            float x = input[i];
            float result = 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
            output[i] = result;
        }
    }
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    cl_int err;

    cl_uint num_platforms;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    checkError(err, "clGetPlatformIDs");

    if (num_platforms == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        exit(1);
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (platform >= static_cast<int>(num_platforms)) {
        throw std::runtime_error("Platform index out of range");
    }

    cl_platform_id cl_platform = platforms[platform];

    cl_uint num_devices;
    err = clGetDeviceIDs(cl_platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    checkError(err, "clGetDeviceIDs (for num_devices)");

    if (num_devices == 0) {
        std::cerr << "No GPU devices found." << std::endl;
        exit(1);
    }

    cl_device_id device;
    err = clGetDeviceIDs(cl_platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    checkError(err, "clGetDeviceIDs (for device)");

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkError(err, "clCreateCommandQueueWithProperties");

    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
    checkError(err, "clGetDeviceInfo (CL_DEVICE_MAX_WORK_GROUP_SIZE)");

    size_t local_work_size = std::min(max_work_group_size, input.size());
    size_t global_work_size = (input.size() + local_work_size - 1) / local_work_size * local_work_size;

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), (void*)input.data(), &err);
    checkError(err, "clCreateBuffer (input)");

    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input.size() * sizeof(float), nullptr, &err);
    checkError(err, "clCreateBuffer (output)");

    const char* kernel_source = gelu_kernel_code.c_str();
    size_t source_size = gelu_kernel_code.size();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, &source_size, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "OpenCL Program Build Error: " << log.data() << std::endl;
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu", &err);
    checkError(err, "clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    checkError(err, "clSetKernelArg (input)");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    checkError(err, "clSetKernelArg (output)");

    unsigned int size = input.size();
    err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &size);
    checkError(err, "clSetKernelArg (size)");

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    checkError(err, "clEnqueueNDRangeKernel");

    std::vector<float> output(input.size());
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, input.size() * sizeof(float), output.data(), 0, nullptr, nullptr);
    checkError(err, "clEnqueueReadBuffer");

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
