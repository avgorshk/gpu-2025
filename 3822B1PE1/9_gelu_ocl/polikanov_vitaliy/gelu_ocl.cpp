#include "gelu_ocl.h"
#include <CL/opencl.h>
#include <vector>
#include <cstdlib>

const char* gelu_kernel_code = R"(
__kernel void gelu_transform(__global const float* data_in,
                             __global float* data_out,
                             int element_count) {
    int position = get_global_id(0);
    if (position < element_count) {
        float value = data_in[position];
        float cubic_term = 0.044715f * value * value * value;
        float param = 0.7978845608f * (value + cubic_term);
        float exponent = exp(2.0f * param);
        float hyperbolic = (exponent - 1.0f) / (exponent + 1.0f);
        data_out[position] = value * 0.5f * (1.0f + hyperbolic);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input_data,
                                     int selected_platform = 0) {
    size_t data_size = input_data.size();
    std::vector<float> output_data(data_size);
    if (data_size == 0) {
        return output_data;
    }
    cl_platform_id target_platform;
    cl_uint available_platforms;
    clGetPlatformIDs(0, nullptr, &available_platforms);

    cl_platform_id* platform_list = new cl_platform_id[available_platforms];
    clGetPlatformIDs(available_platforms, platform_list, nullptr);

    target_platform = platform_list[selected_platform % available_platforms];
    delete[] platform_list;

    cl_device_id gpu_device;
    cl_uint device_count;
    clGetDeviceIDs(target_platform, CL_DEVICE_TYPE_GPU, 1,
                   &gpu_device, &device_count);
    cl_int status;
    cl_context ocl_context = clCreateContext(nullptr, 1, &gpu_device,
                                             nullptr, nullptr, &status);

    cl_command_queue command_queue = clCreateCommandQueue(ocl_context,
                                                          gpu_device,
                                                          0, &status);
    size_t memory_bytes = data_size * sizeof(float);
    cl_mem input_buffer = clCreateBuffer(ocl_context,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         memory_bytes,
                                         (void*)input_data.data(),
                                         &status);
    cl_mem output_buffer = clCreateBuffer(ocl_context,
                                          CL_MEM_WRITE_ONLY,
                                          memory_bytes,
                                          nullptr,
                                          &status);
    cl_program kernel_program = clCreateProgramWithSource(ocl_context,
                                                          1,
                                                          &gelu_kernel_code,
                                                          nullptr,
                                                          &status);
    clBuildProgram(kernel_program, 1, &gpu_device, nullptr, nullptr, nullptr);
    cl_kernel gelu_kernel = clCreateKernel(kernel_program,
                                           "gelu_transform",
                                           &status);
    clSetKernelArg(gelu_kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(gelu_kernel, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(gelu_kernel, 2, sizeof(int), &data_size);
    size_t workgroup_dim = 256;
    size_t total_work_size = ((data_size + workgroup_dim - 1) / workgroup_dim)
                             * workgroup_dim;
    size_t local_work_size = workgroup_dim;
    clEnqueueNDRangeKernel(command_queue, gelu_kernel, 1,
                           nullptr, &total_work_size, &local_work_size,
                           0, nullptr, nullptr);
    clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE,
                        0, memory_bytes, output_data.data(),
                        0, nullptr, nullptr);
    clFinish(command_queue);
    clReleaseKernel(gelu_kernel);
    clReleaseProgram(kernel_program);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(ocl_context);
    return output_data;
}
