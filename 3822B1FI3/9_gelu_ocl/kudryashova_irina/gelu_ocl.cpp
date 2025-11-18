#include "gelu_ocl.h"
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static const char *kGeluKernelSource = R"CLC(
__kernel void gelu_ocl(__global const float* input,
                       __global float* output,
                       const int size)
{
    int index = get_global_id(0);
    if (index >= size) return;

    float x = input[index];
    const float sqrt_pi = sqrt(2.0f / acos(-1.0f));
    float x2 = x * x;
    float x3 = x2 * x;

    float arg_tanh = sqrt_pi * (x + 0.044715f * x3);
    float exp_term = exp(2.0f * arg_tanh);
    float t = (exp_term - 1.0f) / (exp_term + 1.0f);

    output[index] = 0.5f * x * (1.0f + t);
}
)CLC";

static void CheckCl(cl_int err, const char *msg)
{
    if (err != CL_SUCCESS)
        throw std::runtime_error(std::string("OpenCL: ") + msg + " (" + std::to_string(err) + ")");
}

namespace
{
struct OclState
{
    bool initialized = false;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
} g_state;

void InitOclOnce(int platformIndex)
{
    if (g_state.initialized)
        return;
    cl_int err = CL_SUCCESS;

    cl_uint numPlatforms = 0;
    CheckCl(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs(count)");
    if (numPlatforms == 0 || platformIndex < 0 || (cl_uint)platformIndex >= numPlatforms)
        throw std::runtime_error("Bad OpenCL platform index");

    std::vector<cl_platform_id> platforms(numPlatforms);
    CheckCl(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr), "clGetPlatformIDs(list)");
    g_state.platform = platforms[platformIndex];

    CheckCl(clGetDeviceIDs(g_state.platform, CL_DEVICE_TYPE_GPU, 1, &g_state.device, nullptr), "clGetDeviceIDs(GPU)");

    g_state.context = clCreateContext(nullptr, 1, &g_state.device, nullptr, nullptr, &err);
    CheckCl(err, "clCreateContext");

    g_state.queue = clCreateCommandQueue(g_state.context, g_state.device, 0, &err);
    CheckCl(err, "clCreateCommandQueue");

    const char *src = kGeluKernelSource;
    size_t srcLength = std::strlen(kGeluKernelSource);
    g_state.program = clCreateProgramWithSource(g_state.context, 1, &src, &srcLength, &err);
    CheckCl(err, "clCreateProgramWithSource");

    err = clBuildProgram(g_state.program, 1, &g_state.device, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t logSize = 0;
        clGetProgramBuildInfo(g_state.program, g_state.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(g_state.program, g_state.device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        throw std::runtime_error("clBuildProgram failed:\n" + log);
    }

    g_state.kernel = clCreateKernel(g_state.program, "gelu_ocl", &err);
    CheckCl(err, "clCreateKernel");
    g_state.initialized = true;
}
} // namespace

std::vector<float> GeluOCL(const std::vector<float> &input, int platform)
{
    if (input.empty())
        return {};

    InitOclOnce(platform);

    const size_t size = input.size();
    const size_t bytes = size * sizeof(float);
    cl_int err = CL_SUCCESS;

    cl_mem inputBuffer = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                                        const_cast<float *>(input.data()), &err);
    CheckCl(err, "clCreateBuffer(input)");

    cl_mem outputBuffer = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    CheckCl(err, "clCreateBuffer(output)");

    CheckCl(clSetKernelArg(g_state.kernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg(0)");
    CheckCl(clSetKernelArg(g_state.kernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg(1)");

    int nInt = static_cast<int>(size);
    CheckCl(clSetKernelArg(g_state.kernel, 2, sizeof(int), &nInt), "clSetKernelArg(2)");

    const size_t localSize = 256;
    const size_t globalSize = ((size + localSize - 1) / localSize) * localSize;

    CheckCl(
        clEnqueueNDRangeKernel(g_state.queue, g_state.kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel");

    std::vector<float> output(size);
    CheckCl(clEnqueueReadBuffer(g_state.queue, outputBuffer, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer");

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    return output;
}
