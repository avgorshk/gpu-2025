#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <cmath>
#include <mutex>

static cl_context g_context = nullptr;
static cl_command_queue g_queue = nullptr;
static cl_kernel g_kernel = nullptr;
static cl_device_id g_device = nullptr;
static int g_initialized_platform = -1;
static std::mutex g_init_mutex;

static const char* gelu_source_code = R"(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float x = input[gid];
        float x3 = x * x * x;
        float inner = 0.79788456f * (x + 0.044715f * x3); // approximation
        float e = exp(inner);
        output[gid] = x * e / (1.0f + e);
    }
}
)";

static void InitOpenCL(int platform_id) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_context && g_initialized_platform == platform_id)
        return;

    g_initialized_platform = platform_id;

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_platform_id plat = platforms[platform_id];

    cl_uint num_devices = 0;
    clGetDeviceIDs(plat, CL_DEVICE_GPU, 0, nullptr, &num_devices);
    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(plat, CL_DEVICE_GPU, num_devices, devices.data(), nullptr);

    g_device = devices[0];
    cl_int err;
    g_context = clCreateContext(nullptr, 1, &g_device, nullptr, nullptr, &err);

    g_queue = clCreateCommandQueueWithProperties(g_context, g_device, 0, &err);

    cl_program program = clCreateProgramWithSource(g_context, 1, &gelu_source_code, nullptr, &err);
    clBuildProgram(program, 1, &g_device, nullptr, nullptr, nullptr);
    g_kernel = clCreateKernel(program, "gelu_kernel", &err);
    clReleaseProgram(program);
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    int n = static_cast<int>(input.size());
    std::vector<float> output(n);
    if (n == 0) return output;

    InitOpenCL(platform);

    cl_int err;
    size_t buf_size = n * sizeof(float);

    cl_mem input_buf = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      buf_size, const_cast<float*>(input.data()), &err);
    cl_mem output_buf = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, buf_size, nullptr, &err);

    clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &output_buf);
    clSetKernelArg(g_kernel, 2, sizeof(int), &n);

    size_t local_work_size = 256;
    size_t global_work_size = ((n + local_work_size - 1) / local_work_size) * local_work_size;

    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &global_work_size, &local_work_size,
                           0, nullptr, nullptr);

    clEnqueueReadBuffer(g_queue, output_buf, CL_TRUE, 0, buf_size, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);

    return output;
}
