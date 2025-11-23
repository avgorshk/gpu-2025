#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>

static const char* gelu_kernel_source = R"(
__kernel void gelu_compute(__global const float* input_data,
                           __global float* output_data,
                           const int size) {
    int gid = get_global_id(0);
    if (gid < size) {
        float val = input_data[gid];
        float val_cubed = val * val * val;
        float inner = 0.7978845608f * (val + 0.044715f * val_cubed);
        float tanh_approx = (exp(2.0f * inner) - 1.0f) / (exp(2.0f * inner) + 1.0f);
        output_data[gid] = 0.5f * val * (1.0f + tanh_approx);
    }
}
)";

static cl_context g_context = nullptr;
static cl_command_queue g_queue = nullptr;
static cl_kernel g_kernel = nullptr;
static int g_platform_id = -1;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    int data_size = static_cast<int>(input.size());

    if (g_platform_id != platform) {
        g_platform_id = platform;

        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);

        std::vector<cl_platform_id> platform_list(platform_count);
        clGetPlatformIDs(platform_count, platform_list.data(), nullptr);
        cl_platform_id selected_platform = platform_list[platform];

        cl_uint device_count = 0;
        clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);

        std::vector<cl_device_id> device_list(device_count);
        clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, device_count, device_list.data(), nullptr);

        cl_int error_code;
        g_context = clCreateContext(nullptr, 1, &device_list[0], nullptr, nullptr, &error_code);
        g_queue = clCreateCommandQueueWithProperties(g_context, device_list[0], 0, &error_code);

        cl_program prog = clCreateProgramWithSource(g_context, 1, &gelu_kernel_source, nullptr, &error_code);
        clBuildProgram(prog, 1, &device_list[0], nullptr, nullptr, nullptr);
        g_kernel = clCreateKernel(prog, "gelu_compute", &error_code);

        clReleaseProgram(prog);
    }

    cl_int status;
    size_t buffer_bytes = data_size * sizeof(float);
    
    cl_mem input_buffer = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         buffer_bytes, const_cast<float*>(input.data()), &status);
    cl_mem output_buffer = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, buffer_bytes, nullptr, &status);

    clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(g_kernel, 2, sizeof(int), &data_size);

    size_t work_group = 256;
    size_t global_work = ((data_size + work_group - 1) / work_group) * work_group;

    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &global_work, &work_group, 0, nullptr, nullptr);

    std::vector<float> result(data_size);
    clEnqueueReadBuffer(g_queue, output_buffer, CL_TRUE, 0, buffer_bytes, result.data(), 0, nullptr, nullptr);

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);

    return result;
}